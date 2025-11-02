import asyncio
import os
from typing import List, Optional, Dict
from dotenv import load_dotenv

#导入 Reranker 模块
from .reranker import RerankerModel
from .light_graph_rag import LightRAG
from .nodes import WorkflowNodes
from .graph import create_indexing_graph, create_querying_graph
from .state import IndexingState
from .utils import logger
from .kg.shared_storage import initialize_pipeline_status
from .llm import get_llm
from .async_lanchain_rag_adapter import create_lightrag_compatible_complete
from .embedding_factory import get_embedder, create_lightrag_embedding_adapter
from .mineru_integration import SmartDocumentIndexer

#加载环境变量
load_dotenv()

# --- RAGAgent with Real LLM and Embedding ---
class RAGAgent:
    def __init__(self):
        self.rag: Optional[LightRAG] = None
        self.nodes: Optional[WorkflowNodes] = None
        self.indexing_graph = None
        self.querying_graph = None
        self.smart_indexer: Optional[SmartDocumentIndexer] = None
        self.reranker: Optional[RerankerModel] = None
        self.langchain_llm = None
        self.rerank_top_k = 20

    @classmethod
    async def create(cls, working_dir: str = "data/rag_storage", rerank_config: dict = None, storage_mode: str = "database"):
        """
        创建RAGAgent实例
        
        Args:
            working_dir: 工作目录路径
            rerank_config: 重排序配置
            storage_mode: 存储模式，可选"memory"（内存管理）或"database"（数据库存储）
        """
        instance = cls()
        instance.working_dir = working_dir
        instance.storage_mode = storage_mode  # 存储存储模式
        os.makedirs(working_dir, exist_ok=True)
        
        # === 1. 获取 LangChain LLM 实例 ===
        langchain_llm = get_llm() 
        
        # LLM 实例
        instance.langchain_llm = langchain_llm
        logger.info(f"✅ LangChain LLM 已加载用于问答生成")
        
        # === 2. 包装为 LightRAG 兼容的异步函数 ===
        llm_func = create_lightrag_compatible_complete(
            langchain_llm,
            retry_attempts=3,
            retry_min_wait=4
        )
        
        # === 3. 获取嵌入模型（从 .env 构建配置） ===
        etype = os.getenv("EMBEDDING_TYPE", "ollama").strip()
        
        if etype == "hf":
            model_name = os.getenv("HF_EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
            model_kwargs = {}
            device = os.getenv("HF_EMBEDDING_DEVICE", "").strip()
            if device:
                model_kwargs["device"] = device
            # 可选附加参数
            if os.getenv("HF_EMBEDDING_TRUST_REMOTE_CODE", "false").lower() == "true":
                model_kwargs["trust_remote_code"] = True
            
            embedding_config = {
                "type": "hf",
                "model_name": model_name,
                "model_kwargs": model_kwargs,
                "encode_kwargs": {},
                "show_progress": os.getenv("HF_EMBEDDING_SHOW_PROGRESS", "false").lower() == "true",
                "multi_process": os.getenv("HF_EMBEDDING_MULTI_PROCESS", "false").lower() == "true",
            }
            logger.info(f"✅ 配置 HuggingFace Embedding: {model_name}")
        
        elif etype == "ollama":
            embedding_config = {
                "type": "ollama",
                "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b").strip(),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
            }
            logger.info(f"✅ 配置 Ollama Embedding: {embedding_config['model']}")
        
        elif etype == "vllm":
            base_url = os.getenv("VLLM_BASE_URL")
            if not base_url:
                raise ValueError("EMBEDDING_TYPE=vllm 但未配置 VLLM_BASE_URL")
            embedding_config = {
                "type": "vllm",
                "model": os.getenv("VLLM_EMBEDDING_MODEL", "text-embedding-3-large").strip(),
                "base_url": base_url.strip(),
                "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
            }
            logger.info(f"✅ 配置 vLLM Embedding: {embedding_config['model']}")
        else:
            raise ValueError(f"未知 EMBEDDING_TYPE: {etype}，支持: hf, ollama, vllm")
        
        # 创建 LangChain 嵌入模型实例
        langchain_embedder = get_embedder(embedding_config)
        
        # 从 .env 读取 embedding 维度
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        
        # 适配为 LightRAG 兼容的嵌入函数
        embedding_func = create_lightrag_embedding_adapter(
            langchain_embedder,
            embedding_dim=embedding_dim
        )
        
        # === 4. 初始化 Reranker 模型（.env 为默认，入参可覆盖） ===
        env_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        cfg = rerank_config or {}
        enabled = cfg.get("enabled", env_enabled)
        
        if enabled:
            logger.info("🔧 初始化 Reranker 模型...")
            try:
                model = cfg.get("model", os.getenv("RERANK_MODEL", "maidalun1020/bce-reranker-base_v1").strip())
                device = cfg.get("device", os.getenv("RERANK_DEVICE", "").strip() or None)
                top_k = int(cfg.get("top_k", os.getenv("RERANK_TOP_K", "20")))
                use_fp16 = cfg.get("use_fp16", os.getenv("RERANK_USE_FP16", "false").lower() == "true")
                
                instance.reranker = RerankerModel(
                    model_name_or_path=model,
                    device=device,
                    top_k=top_k,
                    use_fp16=use_fp16,
                )
                instance.rerank_top_k = top_k
                logger.info(f"✅ Reranker 模型加载完成 (model={model}, top_k={instance.rerank_top_k})")
            except Exception as e:
                logger.error(f"❌ 加载 Reranker 模型失败: {e}")
                instance.reranker = None
        
        # === 5. 创建 LightRAG 实例 ===
        # 根据storage_mode参数选择不同的存储方式
        if storage_mode == "memory":
            # 内存管理模式 - 使用本地JSON文件存储
            logger.info("📝 使用内存管理模式 - 本地JSON文件存储")
            instance.rag = LightRAG( 
                working_dir=working_dir,
                embedding_func=embedding_func,
                llm_model_func=llm_func,
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage",
                doc_status_storage="JsonDocStatusStorage"
            )
        else:
            # 数据库存储模式 - 使用PostgreSQL存储（默认）
            logger.info("🗄️ 使用数据库存储模式 - PostgreSQL存储")
            instance.rag = LightRAG( 
                working_dir=working_dir,
                embedding_func=embedding_func,
                llm_model_func=llm_func,
                kv_storage="PGKVStorage",
                vector_storage="PGVectorStorage",
                graph_storage="PGGraphStorage",
                doc_status_storage="PGDocStatusStorage"
            )
        
        # === 6. 初始化存储和流水线 ===
        await instance.rag.initialize_storages()
        await initialize_pipeline_status()
        
        # === 7. 初始化工作流 ===
        instance.nodes = WorkflowNodes(instance.rag)
        instance.indexing_graph = create_indexing_graph(instance.nodes)
        instance.querying_graph = create_querying_graph(instance.nodes)
        
        # === 8. 初始化智能文档索引器 ===
        # 从环境变量获取MinerU API密钥
        mineru_api_key = os.environ.get("MINERU_API_KEY", "")
        instance.smart_indexer = SmartDocumentIndexer(mineru_api_key=mineru_api_key)
        
        return instance

    async def index_documents(self, file_paths: List[str]):
        """智能索引文档 - 支持PDF和文本文件"""
        logger.info(f"📚 开始智能索引 {len(file_paths)} 个文档...")
        
        # 使用智能文档索引器处理文件
        if self.smart_indexer:
            process_result = await self.smart_indexer.process_files_for_indexing(file_paths)
            files_to_index = process_result["files_to_index"]
            
            if not files_to_index:
                logger.warning("没有可索引的文件")
                return {
                    "track_id": None,
                    "status_message": "没有可索引的文件",
                    "processing_summary": self.smart_indexer.get_processing_summary(process_result)
                }
            
            logger.info(f"📄 准备索引 {len(files_to_index)} 个处理后的文件")
        else:
            # 如果没有智能索引器，直接使用原始文件
            files_to_index = file_paths
        
        # 读取文件内容进行索引
        contents, ids, paths = [], [], []
        for fp in files_to_index:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    contents.append(f.read())
                ids.append(os.path.basename(fp))
                paths.append(os.path.abspath(fp))
                logger.info(f"📖 读取文件: {os.path.basename(fp)}")
            except Exception as e:
                logger.error(f"❌ 读取文件失败 {fp}: {e}")
                continue
        
        if not contents:
            logger.error("没有成功读取任何文件内容")
            return {
                "track_id": None,
                "status_message": "文件读取失败",
                "processing_summary": "文件读取失败"
            }

        initial_state: IndexingState = {
            "working_dir": self.working_dir,
            "inputs": contents,
            "ids": ids,
            "file_paths": paths,
            "track_id": None,
            "status_message": ""
        }

        result = await self.indexing_graph.ainvoke(initial_state)
        
        # 添加处理摘要到结果中
        if self.smart_indexer and 'processing_summary' not in result:
            result['processing_summary'] = self.smart_indexer.get_processing_summary(process_result)
        
        logger.info(f"📌 索引流程结束: {result['status_message']}")
        return result

    async def query(
        self, 
        question: str, 
        mode: str = "mix", 
        enable_rerank: bool = True,
        rerank_top_k: Optional[int] = None,
        chat_history: List[Dict] = None,
        thread_id: str = None
    ):
        """通过 LangGraph 查询流程查询知识图谱
        
        Args:
            question: 查询问题
            mode: 查询模式 (naive, local, global, hybrid, mix)
            enable_rerank: 是否启用精排
            rerank_top_k: 精排数量
            chat_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            thread_id: 会话标识（可选，用于会话管理）
        
        Returns:
            包含 context, answer, chat_history 的字典
        """
        from src.Knowledge_Graph_Agent.state import QueryState

        # 如果未提供 thread_id，生成一个临时 ID
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())

        initial_query_state: QueryState = {
            "thread_id": thread_id,  # 传入会话标识
            "working_dir": self.working_dir,
            "query": question,
            "query_mode": mode,
            "llm": self.langchain_llm,
            "reranker": self.reranker if enable_rerank else None,
            "rerank_top_k": self.rerank_top_k,
            "chat_history": chat_history or [],
            "retrieved_docs": [],
            "retrieved_entities": [],
            "retrieved_relationships": [],
            "final_docs": [],
            "context": {},
            "answer": ""
        }

        # 通过 config 传递 thread_id（用于 LangGraph 内部追踪）
        config = {"configurable": {"thread_id": thread_id}}

        # ainvoke 是独立执行，状态不会跨调用保留
        result = await self.querying_graph.ainvoke(
            initial_query_state,
            config=config  # 可用于检查点/持久化
        )

        logger.info(f"🔍 查询流程完成 (thread_id={thread_id[:8]}..., mode={mode})")

        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", {}),
            "chat_history": result.get("chat_history", [])
        }


# --- Main ---
async def main():
    """示例用法"""
    agent = await RAGAgent.create()
    
    # 索引文档
    await agent.index_documents(["data/inputs/111002_tk.md"])
    
    # 第一轮查询
    result1 = await agent.query(
        "这份保险条款的主要内容是什么?", 
        mode="hybrid",
        enable_rerank=True
    )
    print("\n🤖 第一轮答案:", result1["answer"])
    
    # 第二轮查询（带对话历史）
    result2 = await agent.query(
        "那犹豫期是多长时间?",
        mode="hybrid", 
        enable_rerank=True,
        chat_history=result1["chat_history"]  # 传入对话历史
    )
    print("\n🤖 第二轮答案:", result2["answer"])


if __name__ == "__main__":
    asyncio.run(main())