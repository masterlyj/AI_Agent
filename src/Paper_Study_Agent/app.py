import os
from typing import List, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .graph import load_and_chunk_papers, embed_and_index, retrieve, generate_answer, update_convstore
from .state import Paper_Study_State
from .embedding_factory import get_embedder
from .llm import get_llm
from .graph import retrieve, rerank, generate_answer # 导入新节点
from .reranker import RerankerModel # 导入 RerankerModel 和配置

load_dotenv()


class PaperChatBot:
    def __init__(self, arxiv_ids: List[str], embedding_config: Optional[dict] = None, rerank_config: Optional[dict] = None):
        """
        初始化论文聊天机器人
        
        Args:
            arxiv_ids: arXiv 论文 ID 列表
            embedding_config: Embedding 配置（可选，未提供时从 .env 读取）
            rerank_config: Rerank 配置（可选，未提供时从 .env 读取）
        """
        self.arxiv_ids = arxiv_ids
        
        # === Embedding: 优先使用入参；否则从 .env 构建 ===
        if not embedding_config:
            etype = os.getenv("EMBEDDING_TYPE", "ollama").strip()
            
            if etype == "hf":
                model_name = os.getenv("HF_EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip()
                model_kwargs = {}
                device = os.getenv("HF_EMBEDDING_DEVICE", "").strip()
                if device:
                    model_kwargs["device"] = device
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
                print(f"✅ 配置 HuggingFace Embedding: {model_name}")
            
            elif etype == "ollama":
                embedding_config = {
                    "type": "ollama",
                    "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b").strip(),
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
                }
                print(f"✅ 配置 Ollama Embedding: {embedding_config['model']}")
            
            elif etype == "vllm":
                base_url = os.getenv("VLLM_BASE_URL")
                if not base_url:
                    raise ValueError("EMBEDDING_TYPE=vllm 但未配置 VLLM_BASE_URL")
                embedding_config = {
                    "type": "vllm",
                    "model": os.getenv("VLLM_EMBEDDING_MODEL", "text-embedding-3-large").strip(),
                    "base_url": base_url,
                    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
                }
                print(f"✅ 配置 vLLM Embedding: {embedding_config['model']}")
            else:
                raise ValueError(f"未知 EMBEDDING_TYPE: {etype}")
        
        self.embedder = get_embedder(embedding_config)
        self.llm = get_llm()
        
        # === Rerank: 优先用入参；否则看 .env ===
        self.reranker = None
        env_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        cfg = rerank_config or {}
        enabled = cfg.get("enabled", env_enabled)
        
        if enabled:
            print("🚀 正在加载 Reranker 模型...")
            model = cfg.get("model", os.getenv("RERANK_MODEL", "maidalun1020/bce-reranker-base_v1").strip())
            device = cfg.get("device", os.getenv("RERANK_DEVICE", "").strip() or None)
            top_k = int(cfg.get("top_k", os.getenv("RERANK_TOP_K", "20")))
            use_fp16 = cfg.get("use_fp16", os.getenv("RERANK_USE_FP16", "false").lower() == "true")
            
            self.reranker = RerankerModel(
                model_name_or_path=model,
                device=device,
                top_k=top_k,
                use_fp16=use_fp16
            )
            print(f"✅ Reranker 模型加载完成 (model={model}, top_k={top_k})")
        
        print("🚀 正在初始化论文向量库，请稍候...")
        
        # --- 初始化流程：手动执行节点以构建向量库 ---
        init_temp_state: Paper_Study_State = {
            "arXiv_ids": arxiv_ids,
            "query": "",
            "context": [],
            "embedder": self.embedder,
            "vectorstore": None,
            "convstore": None,
            "context_retrieved": "",
            "history_retrieved": "",
            "answer": "",
            "messages": [],
        }
        
        # 手动调用 load_and_chunk_papers
        load_command = load_and_chunk_papers(init_temp_state)
        if load_command.update:
            init_temp_state.update(load_command.update)
        
        # 手动调用 embed_and_index
        embed_command = embed_and_index(init_temp_state)
        if embed_command.update:
            init_temp_state.update(embed_command.update)
        
        self.base_vectorstore = init_temp_state["vectorstore"]
        self.convstore = init_temp_state["convstore"]

        # --- 主聊天图定义 ---
        workflow = StateGraph(Paper_Study_State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rerank", rerank) # 新增 rerank 节点
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("update_convstore", update_convstore)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate_answer")
        workflow.add_edge("generate_answer", "update_convstore")
        workflow.add_edge("update_convstore", END)

        self.graph = workflow.compile()

        # 构建初始消息
        doc_summary = "可用论文列表：\n"
        for doc in init_temp_state["context"]:
            if doc.metadata.get("type") == "global_context":
                doc_summary = doc.page_content
                break
        self.initial_msg = (
            "你好！我是一个文档聊天助手，旨在为用户提供帮助！\n"
            f"{doc_summary}\n\n我能为您提供什么帮助？"
        )

    def chat(self, message: str, history: List[List[str]]) -> str:
        current_state: Paper_Study_State = {
            "arXiv_ids": self.arxiv_ids,
            "query": message,
            "context": [],
            "embedder": self.embedder,
            "vectorstore": self.base_vectorstore,
            "convstore": self.convstore,
            "context_retrieved": "",
            "history_retrieved": "",
            "answer": "",
            "messages": [],
            "reranker": self.reranker,  # 传递 Reranker 模型实例
        }

        result = self.graph.invoke(current_state)
        self.convstore = result["convstore"]
        return result["answer"]

    def get_initial_message(self) -> str:
        return self.initial_msg