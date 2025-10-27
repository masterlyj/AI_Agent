from typing import Dict, Any, TypedDict, Literal, Optional, List
from pathlib import Path
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
            query_mode = state.get("query_mode", "mix")
            
            logger.info(f"正在以 '{query_mode}' 模式为查询进行粗排检索...")
            retrieval_result = await self.rag.aquery_data(
                query,
                param=QueryParam(mode=query_mode, chunk_top_k=40)
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
            
            # 从 reranker 配置中获取 top_k，如果没有则默认为 20
            top_k = getattr(reranker, 'rerank_top_k', 20)
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

    async def generate_answer(self, state: "QueryState") -> Dict[str, Any]:
        """
        节点: 基于精排后的上下文生成最终答案。
        使用 LangChain LLM，支持多轮对话。
        """
        logger.info("--- 运行节点:generate_answer (生成答案) ---")
        try:
            query = state["query"]
            final_docs = state.get("final_docs", [])
            retrieved_entities = state.get("retrieved_entities", [])
            retrieved_relationships = state.get("retrieved_relationships", [])
            chat_history = state.get("chat_history", [])  # 获取对话历史
            llm = state.get("llm")  # 获取 LLM 实例
            
            if not llm:
                raise ValueError("❌ LLM 实例未在 state 中配置")
            
            if not final_docs and not retrieved_entities and not retrieved_relationships:
                logger.warning("⚠️ 没有上下文可供生成答案。")
                return {
                    "answer": "抱歉,根据可用信息我无法回答您的问题。",
                    "chat_history": chat_history + [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": "抱歉,根据可用信息我无法回答您的问题。"}
                    ],
                    "context": {
                        "raw_context": "",
                        "query_mode": state.get("query_mode", "hybrid"),
                    }
                }

            # ==================== 标准化知识图谱上下文 ====================
            kg_context = self._format_knowledge_graph(retrieved_entities, retrieved_relationships)
            
            # ==================== 标准化文档上下文 ====================
            doc_context = self._format_documents(final_docs)
            
            # ==================== 构建完整上下文 ====================
            full_context = ""
            if kg_context:
                full_context += "# 📊 知识图谱信息\n\n" + kg_context + "\n"
            if doc_context:
                full_context += "# 📄 相关条款文档\n\n" + doc_context
            
            # ==================== 构建保险领域专用提示词 ====================
            system_prompt = """你是一位专业的保险咨询顾问，擅长解读保险条款、理赔规则和产品说明。

**你的职责:**
1. 基于知识图谱中的实体关系理解保险业务逻辑
2. 结合文档原文提供准确的条款解释
3. 用清晰易懂的语言解答客户疑问

**回答原则:**
- **准确性优先**: 严格依据提供的保险条款和知识图谱
- **结构化表达**: 使用分点、分段的方式组织答案
- **引用来源**: 在关键信息后标注来源实体或条款
- **风险提示**: 涉及免责条款时需特别强调
- **诚实表达**: 信息不足时明确告知,不可臆测

**回答格式建议:**
1. 直接回答核心问题
2. 列举关键条款和依据
3. 补充注意事项或限制条件
"""
            
            # ==================== 构建消息列表（包含对话历史）====================
            logger.info("🤖 开始调用 LangChain LLM 生成答案...")
            logger.info(f"📊 上下文统计:")
            logger.info(f"   - 实体: {len(retrieved_entities)} 个")
            logger.info(f"   - 关系: {len(retrieved_relationships)} 条")
            logger.info(f"   - 文档: {len(final_docs)} 个")
            logger.info(f"   - 对话历史: {len(chat_history)} 轮")
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加对话历史
            for turn in chat_history:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
            
            # 添加当前查询（包含上下文）
            user_message = f"""请基于以下保险知识库信息回答问题:

{full_context}

**用户问题:** {query}"""
            
            messages.append({"role": "user", "content": user_message})
            
            # 调用 LangChain LLM
            try:
                response = await llm.ainvoke(messages)
                answer = response.content
            except Exception as e:
                logger.error(f"❌ LLM 调用失败: {e}")
                answer = f"生成答案时出错: {str(e)}"
            
            # 🆕 更新对话历史
            new_history = chat_history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ]
            
            logger.info(f"✅ 答案生成完成 (长度: {len(answer)} 字符)")
            
            return {
                "answer": answer,
                "chat_history": new_history,  # 返回更新后的对话历史
                "context": {
                    "raw_context": full_context,
                    "query_mode": state.get("query_mode", "hybrid"),
                    "num_docs_used": len(final_docs),
                    "num_entities": len(retrieved_entities),
                    "num_relationships": len(retrieved_relationships),
                    "rerank_enabled": state.get("reranker") is not None,
                    "entities": retrieved_entities,  # 传递给前端可视化
                    "relationships": retrieved_relationships,  # 传递给前端可视化
                    "documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        } for doc in final_docs
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "answer": f"生成答案时出错: {str(e)}",
                "chat_history": chat_history,
                "context": {}
            }

    def _format_knowledge_graph(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """标准化格式化知识图谱上下文"""
        if not entities and not relationships:
            return ""
        
        kg_parts = []
        
        # 格式化实体
        if entities:
            kg_parts.append("## 🏷️ 相关实体\n")
            for idx, entity in enumerate(entities[:10], 1):  # 限制前10个
                name = entity.get('entity_name', '未知')
                type_ = entity.get('entity_type', '未知类型')
                desc = entity.get('description', '无描述')
                
                kg_parts.append(f"**[{idx}] {name}** `{type_}`\n")
                kg_parts.append(f"  └─ {desc}\n\n")
        
        # 格式化关系
        if relationships:
            kg_parts.append("## 🔗 实体关系\n")
            for idx, rel in enumerate(relationships[:10], 1):  # 限制前10条
                src = rel.get('src_id', '?')
                tgt = rel.get('tgt_id', '?')
                desc = rel.get('description', '无描述')
                weight = rel.get('weight', 0)
                
                kg_parts.append(f"**[{idx}]** {src} ➜ {tgt} `权重:{weight:.2f}`\n")
                kg_parts.append(f"  └─ {desc}\n\n")
        
        return "".join(kg_parts)

    def _format_documents(self, documents: List[Document]) -> str:
        """标准化格式化文档上下文"""
        if not documents:
            return ""
        
        doc_parts = []
        
        for idx, doc in enumerate(documents, 1):
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            score_str = f"{rerank_score:.4f}" if isinstance(rerank_score, float) else str(rerank_score)
            chunk_id = doc.metadata.get('chunk_id', '未知')
            file_path = doc.metadata.get('file_path', '未知来源')
            
            doc_parts.append(f"### 📑 文档片段 {idx}\n")
            doc_parts.append(f"- **来源文件:** {Path(file_path).name}\n")
            doc_parts.append(f"- **片段ID:** {chunk_id}\n")
            doc_parts.append(f"- **相关度评分:** {score_str}\n")
            doc_parts.append(f"\n**内容:**\n```\n{doc.page_content}\n```\n\n")
            doc_parts.append("---\n\n")
        
        return "".join(doc_parts)