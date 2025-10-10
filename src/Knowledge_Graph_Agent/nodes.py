# src/Knowledge_Graph_Agent/nodes.py

from typing import Dict, Any
from .light_graph_rag import LightRAG
from .state import IndexingState, QueryState
from .utils import logger

class WorkflowNodes:
    def __init__(self, rag_instance: LightRAG):
        self.rag = rag_instance

    # === Indexing: 单一节点调用 ainsert ===
    async def index_documents(self, state: IndexingState) -> Dict[str, Any]:
        """
        节点：触发 LightRAG 的完整索引流程。
        将复杂队列逻辑封装在 ainsert 内部，LangGraph 只负责启动和监控。
        """
        logger.info("--- Running Node: index_documents ---")
        try:
            track_id = await self.rag.ainsert(
                input=state["inputs"],
                ids=state.get("ids"),
                file_paths=state.get("file_paths")
            )
            logger.info(f"✅ 索引任务已提交，Track ID: {track_id}")
            return {
                "track_id": track_id,
                "status_message": "Document indexing started successfully."
            }
        except Exception as e:
            logger.error(f"❌ 索引失败: {e}")
            return {
                "track_id": None,
                "status_message": f"Indexing failed: {str(e)}"
            }

    # === Querying: 暂时保留 mock，后续替换为真实逻辑 ===
    async def retrieve_context(self, state: QueryState) -> Dict[str, Any]:
        logger.info("--- Running Node: retrieve_context (mock) ---")
        return {
            "context": {
                "chunks": ["Mock retrieved context for: " + state["query"]],
                "entities": [],
                "relations": []
            }
        }

    async def generate_answer(self, state: QueryState) -> Dict[str, Any]:
        logger.info("--- Running Node: generate_answer (mock) ---")
        return {
            "answer": f"Mock answer to: '{state['query']}'"
        }