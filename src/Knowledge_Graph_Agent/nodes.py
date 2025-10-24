from typing import Dict, Any, TypedDict, Literal, Optional, List
#新增Document导入
from langchain_core.documents import Document
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
            
            # 调用 LightRAG 的 aquery_data 方法，它只检索数据而不调用 LLM
            # 我们检索更多的文档（例如 20 个）以供精排
            logger.info(f"正在以 '{query_mode}' 模式为查询进行粗排检索...")
            retrieval_result = await self.rag.aquery_data(
                query,
                param=QueryParam(mode=query_mode, chunk_top_k=20)
            )
            
            # 从返回的结构化数据中提取所有信息
            data = retrieval_result.get("data", {})
            retrieved_chunks_data = data.get("chunks", [])
            retrieved_entities = data.get("entities", [])
            retrieved_relationships = data.get("relationships", [])
            
            # 将字典格式的 chunks 转换为 LangChain 的 Document 对象，以便后续处理
            retrieved_docs = [
                Document(
                    page_content=chunk.get("content", ""),
                    metadata={
                        "file_path": chunk.get("file_path"),
                        "chunk_id": chunk.get("chunk_id"),
                        "reference_id": chunk.get("reference_id"),
                    }
                ) for chunk in retrieved_chunks_data
            ]
            
            logger.info(f"✅ 粗排检索完成:")
            logger.info(f"   - 文档块: {len(retrieved_docs)} 个")
            logger.info(f"   - 实体: {len(retrieved_entities)} 个")
            logger.info(f"   - 关系: {len(retrieved_relationships)} 条")
            
            # # 打印实体信息
            # if retrieved_entities:
            #     logger.info("\n" + "=" * 60)
            #     logger.info("📊 检索到的实体")
            #     logger.info("=" * 60)
            #     for idx, entity in enumerate(retrieved_entities[:5], 1):  # 只显示前5个
            #         logger.info(f"  [{idx}] {entity.get('entity_name', '未知')}")
            #         logger.info(f"      类型: {entity.get('entity_type', '未知')}")
            #         logger.info(f"      描述: {entity.get('description', '无')[:100]}")
            #     if len(retrieved_entities) > 5:
            #         logger.info(f"  ... 及其他 {len(retrieved_entities) - 5} 个实体")
            #     logger.info("=" * 60 + "\n")
            
            # # 打印关系信息
            # if retrieved_relationships:
            #     logger.info("\n" + "=" * 60)
            #     logger.info("🔗 检索到的关系")
            #     logger.info("=" * 60)
            #     for idx, rel in enumerate(retrieved_relationships[:5], 1):  # 只显示前5条
            #         logger.info(f"  [{idx}] {rel.get('src_id', '?')} → {rel.get('tgt_id', '?')}")
            #         logger.info(f"      关系: {rel.get('description', '无')[:100]}")
            #         logger.info(f"      权重: {rel.get('weight', 0):.2f}")
            #     if len(retrieved_relationships) > 5:
            #         logger.info(f"  ... 及其他 {len(retrieved_relationships) - 5} 条关系")
            #     logger.info("=" * 60 + "\n")
            
            # 将原始文档列表和知识图谱信息放入 state，传递给 rerank 节点
            return {
                "retrieved_docs": retrieved_docs,
                "retrieved_entities": retrieved_entities,
                "retrieved_relationships": retrieved_relationships
            }
            
        except Exception as e:
            logger.error(f"❌ 上下文检索失败: {e}")
            return {
                "retrieved_docs": [],
                "retrieved_entities": [],
                "retrieved_relationships": []
            }
        
    async def rerank_context(self, state: QueryState) -> Dict[str, Any]:
        """
        节点: 使用 BCE Reranker 对检索到的文档进行精排，并打印分数。
        """
        logger.info("--- 运行节点：rerank_context (精排) ---")

        # # 🔧 详细调试日志
        # logger.info(f"📦 精排节点接收到的 State 信息:")
        # logger.info(f"   - State keys: {list(state.keys())}")
        # logger.info(f"   - 'reranker' in state: {'reranker' in state}")
        # logger.info(f"   - state.get('reranker'): {state.get('reranker')}")
        # logger.info(f"   - type(state.get('reranker')): {type(state.get('reranker'))}")
        # logger.info(f"   - 'retrieved_docs' in state: {'retrieved_docs' in state}")
        # logger.info(f"   - retrieved_docs 数量: {len(state.get('retrieved_docs', []))}")

        reranker = state.get("reranker")
        docs_to_rerank = state.get("retrieved_docs", [])

        # 🔧 增强判断逻辑
        if reranker is None:
            logger.warning("⚠️ Reranker 未配置 (state['reranker'] is None)，跳过精排步骤。")
            return {"final_docs": docs_to_rerank}  # 直接将原始文档传递下去
        
        if not docs_to_rerank:
            logger.warning("⚠️ 没有检索到文档 (retrieved_docs 为空)，跳过精排步骤。")
            return {"final_docs": []}

        try:
            query = state["query"]
            passages = [doc.page_content for doc in docs_to_rerank]
            
            logger.info(f"🎯 开始精排: 对 {len(passages)} 个文档进行重新排序...")
            
            # 调用 reranker 模型进行计算
            results = reranker.rerank(query, passages)
            rerank_ids = results.get('rerank_ids', [])
            rerank_scores = results.get('rerank_scores', [])
            
            if not rerank_ids:
                logger.warning("⚠️ Reranker 未返回有效结果，使用原始文档。")
                return {"final_docs": docs_to_rerank}
            
            reranked_docs = [docs_to_rerank[i] for i in rerank_ids]

            # --- 打印排序结果 ---
            logger.info("\n" + "=" * 60)
            logger.info("🎯 Reranker 打分结果 (置信度从高到低)")
            logger.info("=" * 60)
            for idx, (doc, score) in enumerate(zip(reranked_docs, rerank_scores), 1):
                # 将 rerank 分数添加到元数据中，方便追踪
                doc.metadata['rerank_score'] = score
                content_snippet = doc.page_content[:100].replace("\n", " ")
                if len(doc.page_content) > 100:
                    content_snippet += "..."
                logger.info(f"  [{idx}] 分数: {score:.4f} | 来源: {doc.metadata.get('file_path', '未知')}")
                logger.info(f"      内容: {content_snippet}")
            logger.info("=" * 60 + "\n")
            
            # 从 reranker 配置中获取 top_k，如果没有则默认为 3
            top_k = getattr(reranker, 'rerank_top_k', 3)
            final_docs = reranked_docs[:top_k]
            
            logger.info(f"✅ 精排完成，选取 Top {len(final_docs)} 文档传递给生成节点。")
            
            # 将精排后的最终文档列表放入 state
            return {"final_docs": final_docs}
            
        except Exception as e:
            logger.error(f"❌ 精排过程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("⚠️ 精排失败，使用原始检索文档。")
            return {"final_docs": docs_to_rerank}

    # --- 步骤3: 修改 generate_answer 节点，使其只负责生成 ---
    async def generate_answer(self, state: QueryState) -> Dict[str, Any]:
        """
        节点: 基于精排后的上下文生成最终答案。
        这个节点不再执行任何检索。
        """
        logger.info("--- 运行节点：generate_answer (生成答案) ---")
        try:
            query = state["query"]
            # 从 state 中获取由 rerank 节点提供的最终文档
            final_docs = state.get("final_docs", [])
            # 获取实体和关系信息
            retrieved_entities = state.get("retrieved_entities", [])
            retrieved_relationships = state.get("retrieved_relationships", [])
            
            if not final_docs and not retrieved_entities and not retrieved_relationships:
                logger.warning("⚠️ 没有上下文可供生成答案。")
                return {
                    "answer": "抱歉，根据可用信息我无法回答您的问题。",
                    "context": {
                        "raw_context": "",
                        "query_mode": state.get("query_mode", "hybrid"),
                    }
                }

            # 构建知识图谱上下文（实体和关系）
            kg_context_parts = []
            
            # 添加实体信息
            if retrieved_entities:
                entity_context = "### 相关实体\n\n"
                for idx, entity in enumerate(retrieved_entities[:10], 1):  # 限制前10个
                    entity_name = entity.get('entity_name', '未知')
                    entity_type = entity.get('entity_type', '未知')
                    description = entity.get('description', '无描述')
                    entity_context += f"{idx}. **{entity_name}** ({entity_type})\n   {description}\n\n"
                kg_context_parts.append(entity_context)
            
            # 添加关系信息
            if retrieved_relationships:
                relation_context = "### 相关关系\n\n"
                for idx, rel in enumerate(retrieved_relationships[:10], 1):  # 限制前10条
                    src = rel.get('src_id', '?')
                    tgt = rel.get('tgt_id', '?')
                    desc = rel.get('description', '无描述')
                    weight = rel.get('weight', 0)
                    relation_context += f"{idx}. {src} → {tgt} (权重: {weight:.2f})\n   {desc}\n\n"
                kg_context_parts.append(relation_context)
            
            # 将最终文档格式化为高质量的上下文字符串
            doc_context_parts = []
            if final_docs:
                doc_context_parts.append("### 相关文档\n")
                for idx, doc in enumerate(final_docs, 1):
                    rerank_score = doc.metadata.get('rerank_score', 'N/A')
                    score_str = f"{rerank_score:.4f}" if isinstance(rerank_score, float) else str(rerank_score)
                    chunk_id = doc.metadata.get('chunk_id', '未知')
                    file_path = doc.metadata.get('file_path', '未知')
                    
                    doc_context_parts.append(
                        f"【文档 {idx}】\n"
                        f"Chunk ID: {chunk_id}\n"
                        f"来源: {file_path}\n"
                        f"置信度: {score_str}\n"
                        f"内容:\n{doc.page_content}\n"
                    )
            
            # 组合所有上下文
            kg_context_str = "\n".join(kg_context_parts) if kg_context_parts else ""
            doc_context_str = "\n" + ("-" * 60 + "\n").join(doc_context_parts) if doc_context_parts else ""
            
            # 构建完整上下文
            full_context = ""
            if kg_context_str:
                full_context += "## 知识图谱信息\n\n" + kg_context_str + "\n"
            if doc_context_str:
                full_context += "## 文档内容\n" + doc_context_str
            
            # 构建发送给 LLM 的提示词
            system_prompt = f'''你是一个专业的保险文档问答助手。
请根据下面提供的知识图谱信息和文档内容来回答用户的问题。

知识图谱包含了从文档中提取的实体和关系，提供了结构化的知识视图。
文档内容是经过精排的相关文本片段，按相关性从高到低排序。

回答时请：
1. 优先利用知识图谱的结构化信息理解实体间的关系
2. 结合文档内容提供详细的上下文支持
3. 使用清晰、专业的语气
4. 如果可能，引用具体的实体、关系或文档来源
5. 如果信息不足，请直接告知

--- 相关上下文 ---
{full_context}
--- 上下文结束 ---
'''
            
            logger.info("🤖 开始调用 LLM 生成答案...")
            logger.info(f"📊 上下文统计:")
            logger.info(f"   - 实体: {len(retrieved_entities)} 个")
            logger.info(f"   - 关系: {len(retrieved_relationships)} 条")
            logger.info(f"   - 文档: {len(final_docs)} 个")
            
            # 使用 'bypass' 模式调用 aquery_llm，这会跳过 LightRAG 内部的检索
            # 直接将我们的 system_prompt 和 query 发送给 LLM
            result = await self.rag.aquery_llm(
                query,
                param=QueryParam(mode="bypass"),  # 关键: 跳过内部检索
                system_prompt=system_prompt       # 关键: 注入我们的上下文
            )
            
            # 从返回的复杂字典中提取最终答案
            answer = result.get("llm_response", {}).get("content", "生成答案时出错，未收到有效回复。")
            
            logger.info(f"✅ 答案生成完成 (长度: {len(answer)} 字符)")
            
            return {
                "answer": answer,
                "context": {
                    "raw_context": full_context,
                    "query_mode": state.get("query_mode", "hybrid"),
                    "num_docs_used": len(final_docs),
                    "num_entities": len(retrieved_entities),
                    "num_relationships": len(retrieved_relationships),
                    "rerank_enabled": state.get("reranker") is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "answer": f"生成答案时出错: {str(e)}",
                "context": {}
            }