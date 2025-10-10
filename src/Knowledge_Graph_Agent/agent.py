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

# --- Mock Functions ---
class MockEmbeddingFunc:
    def __init__(self, dim: int = 768):
        self.embedding_dim = dim
    async def __call__(self, texts: List[str], **kwargs) -> List[List[float]]:
        logger.info(f"Mock embedding for {len(texts)} texts.")
        return [[0.1] * self.embedding_dim for _ in texts]

# --- RAGAgent with Real LLM ---
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
        
        # === 3. 创建 LightRAG 实例（使用真实 LLM）===
        instance.rag = LightRAG(
            working_dir=working_dir,
            embedding_func=MockEmbeddingFunc(768),  # 嵌入仍用 mock
            llm_model_func=llm_func,  # 👈 关键：注入真实 LLM 函数
            
        )
        
        # === 4. 初始化存储和流水线 ===
        await instance.rag.initialize_storages()
        await initialize_pipeline_status()
        
        # === 5. 初始化工作流 ===
        instance.nodes = WorkflowNodes(instance.rag)
        instance.indexing_graph = create_indexing_graph(instance.nodes)
        instance.querying_graph = create_querying_graph(instance.nodes)
        
        return instance

    async def index_documents(self, file_paths: List[str]):
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

    async def query(self, question: str):
        initial_state = {
            "working_dir": self.working_dir,
            "query": question,
            "query_mode": "hybrid",
            "context": {},
            "answer": ""
        }
        return await self.querying_graph.ainvoke(initial_state)

# --- Main ---
async def main():
    agent = await RAGAgent.create()
    await agent.index_documents(["data/inputs/111002_tk.md"])
    result = await agent.query("这份保险条款的主要内容是什么？")
    print("\n🤖 答案:", result["answer"])

if __name__ == "__main__":
    asyncio.run(main())