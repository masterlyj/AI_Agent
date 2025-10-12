import asyncio
import os
from typing import List, Optional
from .light_graph_rag import LightRAG
from .nodes import WorkflowNodes
from .graph import create_indexing_graph, create_querying_graph
from .state import IndexingState
from .utils import logger
from .kg.shared_storage import initialize_pipeline_status
from .llm import get_llm
from .async_lanchain_rag_adapter import create_lightrag_compatible_complete
from .embedding_factory import get_embedder, create_lightrag_embedding_adapter

# --- RAGAgent with Real LLM and Embedding ---
class RAGAgent:
    def __init__(self):
        self.rag: Optional[LightRAG] = None
        self.nodes: Optional[WorkflowNodes] = None
        self.indexing_graph = None
        self.querying_graph = None

    @classmethod
    async def create(cls, working_dir: str = "data/rag_storage"):
        instance = cls()
        instance.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        # === 1. 获取 LangChain LLM 实例 ===
        langchain_llm = get_llm()  # 自动选择 DeepSeek / Gemini
        
        # === 2. 包装为 LightRAG 兼容的异步函数 ===
        llm_func = create_lightrag_compatible_complete(
            langchain_llm,
            retry_attempts=3,
            retry_min_wait=4
        )
        
        # === 3. 获取嵌入模型 ===
        # 配置 Ollama 嵌入模型 (qwen3_embedding:0.6b)
        embedding_config = {
            "type": "ollama",
            "model": "qwen3-embedding:0.6b",
            "base_url": "http://localhost:11434"
        }
        
        # 创建 LangChain 嵌入模型实例
        langchain_embedder = get_embedder(embedding_config)
        
        # 适配为 LightRAG 兼容的嵌入函数
        embedding_func = create_lightrag_embedding_adapter(
            langchain_embedder,
            embedding_dim=1024
        )
        
        # === 4. 创建 LightRAG 实例 ===
        instance.rag = LightRAG(
            working_dir=working_dir,
            embedding_func=embedding_func,
            llm_model_func=llm_func,
        )
        
        # === 5. 初始化存储和流水线 ===
        await instance.rag.initialize_storages()
        await initialize_pipeline_status()
        
        # === 6. 初始化工作流 ===
        instance.nodes = WorkflowNodes(instance.rag)
        instance.indexing_graph = create_indexing_graph(instance.nodes)
        instance.querying_graph = create_querying_graph(instance.nodes)
        
        return instance

    async def index_documents(self, file_paths: List[str]):
        """索引文档"""
        contents, ids, paths = [], [], []
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                contents.append(f.read())
            ids.append(os.path.basename(fp))
            paths.append(os.path.abspath(fp))

        initial_state: IndexingState = {
            "working_dir": self.working_dir,
            "inputs": contents,
            "ids": ids,
            "file_paths": paths,
            "track_id": None,
            "status_message": ""
        }

        result = await self.indexing_graph.ainvoke(initial_state)
        logger.info(f"📌 索引流程结束: {result['status_message']}")
        return result

    async def query(self, question: str, mode: str = "hybrid"):
        """通过 LangGraph 查询流程查询知识图谱
        
        Args:
            question: 查询问题
            mode: 查询模式 (naive, local, global, hybrid)
        
        Returns:
            包含 context 和 answer 的字典
        """
        from .state import QueryState
        
        # 构造初始查询状态
        initial_query_state: QueryState = {
            "working_dir": self.working_dir,
            "query": question,
            "query_mode": mode,
            "context": {},
            "answer": ""
        }
        
        result = await self.querying_graph.ainvoke(initial_query_state)
        
        logger.info(f"🔍 查询流程完成 (mode={mode})")
        return result


# --- Main ---
async def main():
    """示例用法"""
    agent = await RAGAgent.create()
    
    # 索引文档
    await agent.index_documents(["data/inputs/111002_tk.md"])
    
    # 查询
    result = await agent.query("这份保险条款的主要内容是什么?", mode="hybrid")
    print("\n🤖 答案:", result)


if __name__ == "__main__":
    asyncio.run(main())