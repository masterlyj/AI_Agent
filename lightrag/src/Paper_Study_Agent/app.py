from typing import List
from langgraph.graph import StateGraph, END

from .graph import load_and_chunk_papers, embed_and_index, retrieve, generate_answer, update_convstore
from .state import Paper_Study_State
from .embedding_factory import get_embedder
from .llm import get_llm

class PaperChatBot:
    def __init__(self, arxiv_ids: List[str], embedding_config: dict, use_lightrag: bool = False):
        self.arxiv_ids = arxiv_ids
        self.embedder = get_embedder(embedding_config)
        self.llm = get_llm()
        self.use_lightrag = use_lightrag
        
        print("🚀 正在初始化论文向量库，请稍候...")
        
        # --- 初始化流程：手动执行节点以构建向量库 ---
        init_temp_state: Paper_Study_State = {
            "thread_id": "main",
            "arXiv_ids": arxiv_ids,
            "query": "",
            "context": [],
            "answer": "",
            "embedder": self.embedder,
            "vectorstore": None,
            "convstore": None,
            "knowledge_graph": None,
            "graph_context": "",
            "use_lightrag": self.use_lightrag,
            "context_retrieved": "",
            "history_retrieved": "",
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
        self.knowledge_graph = init_temp_state.get("knowledge_graph")

        # --- 主聊天图定义 ---
        workflow = StateGraph(Paper_Study_State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("update_convstore", update_convstore)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", "update_convstore")
        workflow.add_edge("update_convstore", END)

        self.graph = workflow.compile()

        # 构建初始消息
        doc_summary = "可用论文列表：\n"
        for doc in init_temp_state["context"]:
            if doc.metadata.get("type") == "global_context":
                doc_summary = doc.page_content
                break
        
        lightrag_info = ""
        if self.use_lightrag and self.knowledge_graph:
            lightrag_info = f"\n🧠 知识图谱已启用: {self.knowledge_graph.get_graph_summary()}\n"
        
        self.initial_msg = (
            "你好！我是一个文档聊天助手，旨在为用户提供帮助！\n"
            f"{doc_summary}{lightrag_info}\n我能为您提供什么帮助？"
        )

    def chat(self, message: str, history: List[List[str]]) -> str:
        current_state: Paper_Study_State = {
            "thread_id": "main",
            "arXiv_ids": self.arxiv_ids,
            "query": message,
            "context": [],
            "answer": "",
            "embedder": self.embedder,
            "vectorstore": self.base_vectorstore,
            "convstore": self.convstore,
            "knowledge_graph": self.knowledge_graph,
            "graph_context": "",
            "use_lightrag": self.use_lightrag,
            "context_retrieved": "",
            "history_retrieved": "",
            "messages": [],
        }

        result = self.graph.invoke(current_state)
        self.convstore = result["convstore"]
        return result["answer"]

    def get_initial_message(self) -> str:
        return self.initial_msg