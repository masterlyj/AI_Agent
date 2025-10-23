from typing import List
from langgraph.graph import StateGraph, END

from .graph import load_and_chunk_papers, embed_and_index, retrieve, generate_answer, update_convstore
from .state import Paper_Study_State
from .embedding_factory import get_embedder
from .llm import get_llm
from .graph import retrieve, rerank, generate_answer # 导入新节点
from .reranker import RerankerModel # 导入 RerankerModel 和配置


class PaperChatBot:
    def __init__(self, arxiv_ids: List[str], embedding_config: dict, rerank_config: dict = None):
        self.arxiv_ids = arxiv_ids
        self.embedder = get_embedder(embedding_config)
        self.llm = get_llm()
        # ----------- 新增: 初始化 Reranker 模型 -----------
        self.reranker = None
        if rerank_config:
            print("🚀 正在加载 Reranker 模型...")
            self.reranker = RerankerModel(
                model_name_or_path=rerank_config.get("model", 'maidalun1020/bce-reranker-base_v1'),
                device=rerank_config.get("device", None)
            )
        # -----------------------------------------------------
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