from typing import Dict, Any
from .light_graph_rag import LightRAG
from .state import IndexingState, QueryState
from .utils import logger
from .base import QueryParam

class WorkflowNodes:
    def __init__(self, rag_instance: LightRAG):
        self.rag = rag_instance

    # === Indexing: 单一节点调用 ainsert ===
    async def index_documents(self, state: IndexingState) -> Dict[str, Any]:
        """
        节点:触发 LightRAG 的完整索引流程。
        将复杂队列逻辑封装在 ainsert 内部,LangGraph 只负责启动和监控。
        """
        logger.info("--- 正在运行节点：index_documents ---")
        try:
            track_id = await self.rag.ainsert(
                input=state["inputs"],
                ids=state.get("ids"),
                file_paths=state.get("file_paths")
            )
            logger.info(f"✅ 索引任务已提交,Track ID: {track_id}")
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

    # === Querying ===
    async def retrieve_context(self, state: QueryState) -> Dict[str, Any]:
        """
        节点:从知识图谱检索上下文
        """
        logger.info("--- 运行节点：retrieve_context ---")
        try:
            query = state["query"]
            query_mode = state.get("query_mode", "hybrid")
            
            # 创建查询参数
            query_param = QueryParam(
                mode=query_mode,
                only_need_context=True,  # 只需要上下文,不生成答案
                stream=False,
                top_k=40,
                chunk_top_k=20,
            )
            
            # 调用 LightRAG 查询获取上下文
            context_result = await self.rag.aquery(
                query,
                param=query_param
            )
            
            logger.info(f"✅ 检索到上下文长度: {len(context_result) if isinstance(context_result, str) else 'N/A'}")
            
            return {
                "context": {
                    "raw_context": context_result,
                    "query_mode": query_mode
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 上下文检索失败: {e}")
            return {
                "context": {
                    "error": str(e),
                    "query_mode": state.get("query_mode", "hybrid")
                }
            }

    async def generate_answer(self, state: QueryState) -> Dict[str, Any]:
        """
        节点:基于上下文生成答案
        """
        logger.info("--- 运行节点：生成答案 ---")
        try:
            query = state["query"]
            query_mode = state.get("query_mode", "hybrid")
            
            # 创建查询参数 (完整查询,生成答案)
            query_param = QueryParam(
                mode=query_mode,
                stream=False,
                top_k=40,
                chunk_top_k=20,
            )
            
            # 调用 LightRAG 完整查询
            answer = await self.rag.aquery(
                query,
                param=query_param
            )
            
            logger.info(f"✅ 答案生成完成")
            
            return {
                "answer": answer if isinstance(answer, str) else str(answer)
            }
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            return {
                "answer": f"生成答案时出错: {str(e)}"
            }