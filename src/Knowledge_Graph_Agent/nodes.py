from typing import Dict, Any, TypedDict, Literal, Optional, List, Tuple, Set
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
            logger.info(f"✅ 实体与关系抽取任务已提交,Track ID: {track_id}")
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

            logger.info(f"正在以 '{query_mode}' 模式为查询进行粗排检索...")
            retrieval_result = await self.rag.aquery_data(
                query,
                param=QueryParam(mode=query_mode)
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
        精排后选取 top_k 文档。
        """
        logger.info("--- 运行节点：rerank_context (精排) ---")
        reranker = state.get("reranker")
        docs_to_rerank = state.get("retrieved_docs", [])

        # 增强判断逻辑
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

            for doc, score in zip(reranked_docs, rerank_scores):
                doc.metadata['rerank_score'] = score

            top_k = state.get("rerank_top_k") or getattr(reranker, 'rerank_top_k', 20)

            # 直接选取 top_k
            final_docs = reranked_docs[:top_k]

            logger.info(f"✅ 精排完成，选取 Top {len(final_docs)} 文档传递给生成节点 (用户设置: {state.get('rerank_top_k', '未设置')})。")

            return {"final_docs": final_docs}

        except Exception as e:
            logger.error(f"❌ 精排过程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("⚠️ 精排失败，使用原始检索文档。")
            return {"final_docs": docs_to_rerank}

    async def generate_answer(self, state: QueryState) -> Dict[str, Any]:
        """
        节点: 基于精排后的上下文生成最终答案。
        这个节点不再执行任何检索。
        """
        logger.info("--- 运行节点：generate_answer (生成答案) ---")
        try:
            query = state["query"]
            final_docs = state.get("final_docs", [])
            retrieved_entities = state.get("retrieved_entities", [])
            retrieved_relationships = state.get("retrieved_relationships", [])
            chat_history = state.get("chat_history", [])
            llm = state.get("llm")
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

            full_context = self._build_context(
                final_docs, retrieved_entities, retrieved_relationships
            )

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

            result = await self.rag.aquery_llm(
                query,
                param=QueryParam(mode="bypass"),
                system_prompt=system_prompt
            )

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
                    "rerank_enabled": state.get("reranker") is not None,
                    "entities": retrieved_entities,
                    "relationships": retrieved_relationships,
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
                "context": {}
            }
    
    async def generate_answer_stream(self, state: QueryState):
        """
        流式生成答案的异步生成器。
        先流式输出思考推理过程，然后逐步yield答案的每个token。
        支持二次检索（requery）：如果LLM判断上下文不足，会触发二次检索后继续生成答案。
        
        特殊处理：
        - 如果模型是 deepseek-reasoner，则跳过深度思考阶段，直接生成答案
        """
        try:
            query = state["query"]
            final_docs = state.get("final_docs", [])
            retrieved_entities = state.get("retrieved_entities", [])
            retrieved_relationships = state.get("retrieved_relationships", [])
            llm = state.get("llm")
            chat_history = state.get("chat_history", [])
            model_name = state.get("model_name", "").lower()  # 获取模型名称
            
            # 判断是否为 deepseek-reasoner 模型
            skip_reasoning = "deepseek-reasoner" in model_name or "reasoner" in model_name
            
            if not final_docs and not retrieved_entities and not retrieved_relationships:
                logger.warning("⚠️ 没有上下文可供生成答案。")
                yield {
                    "type": "answer_chunk",
                    "content": "抱歉，根据可用信息我无法回答您的问题。",
                    "done": True
                }
                return

            # 构建完整的上下文（深度思考阶段限制为前20个最相关的项目，避免信息过载）
            full_context = self._build_context(
                final_docs, retrieved_entities, retrieved_relationships, max_items=20
            )
            
            # 先yield上下文信息
            yield {
                "type": "context",
                "context": {
                    "raw_context": full_context,
                    "query_mode": state.get("query_mode", "hybrid"),
                    "num_docs_used": len(final_docs),
                    "num_entities": len(retrieved_entities),
                    "num_relationships": len(retrieved_relationships),
                    "rerank_enabled": state.get("reranker") is not None,
                    "entities": retrieved_entities,
                    "relationships": retrieved_relationships,
                    "documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        } for doc in final_docs
                    ]
                }
            }
            
            # === 深度思考：让LLM看完整文档内容判断上下文是否充分 ===
            # 如果是 deepseek-reasoner 模型，使用其内置推理能力直接生成答案
            if skip_reasoning:
                logger.info(f"⚡ 检测到 {state.get('model_name', 'unknown')} 模型，使用内置推理能力直接生成答案...")
                
                # 构建完整上下文（限制数量）
                full_context = self._build_context(
                    final_docs, retrieved_entities, retrieved_relationships, max_items=20
                )
                
                # 直接构建答案生成的消息
                answer_system_prompt = '''你是一个专业的保险文档问答助手。
请根据下面提供的知识图谱信息和文档内容来回答用户的问题。

**⚠️ 极其重要：不同保险产品的条款差异**
1. **产品独立性**：每个保险产品都有其独立的保险条款，不同产品的条款内容可能完全不同。
   - 例如：「传家宝」保险条款可能不对全残进行赔付
   - 而「传家福」保险条款可能对全残进行赔付
   
2. **严格匹配**：回答问题时，必须严格区分不同的保险产品：
   - 检查文档来源（file_path、chunk_id）确认产品名称
   - 不要将一个产品的条款套用到另一个产品上
   - 如果用户询问的是产品A，绝不能用产品B的条款来回答
   
3. **信息不足时的处理**：
   - 如果当前上下文中没有找到用户询问的具体产品的相关信息
   - 明确告知用户"当前检索的信息不包含该产品的相关条款"
   - 不要猜测或使用其他产品的信息进行推断

**文档格式说明**
文档内容可能包含表格数据，其格式如下：
- `[SOURCE:__TABLE_ENTITY_X__]` 表示该块对应的表格占位符，X为表格编号
- `[CONTEXT]` 后面是与该表格相关的上下文文本
- `[HTML_TABLE]` 后面是真实表格的HTML内容

**回答要求**
请基于提供的上下文，使用清晰、专业的语气回答用户的问题。如果信息不足或不确定，请明确说明。'''

                answer_messages = [{"role": "system", "content": answer_system_prompt}]
                
                # 添加历史对话
                if chat_history:
                    for msg in chat_history[-4:]:
                        answer_messages.append({
                            "role": msg.get("role"),
                            "content": msg.get("content")
                        })
                
                # 添加当前问题和上下文
                answer_user_message = f"""**用户问题：** {query}

**上下文信息：**

{full_context}

请基于以上信息回答用户的问题。"""
                
                answer_messages.append({"role": "user", "content": answer_user_message})
                
                # 流式生成答案，DeepSeek Reasoner会通过reasoning_content字段返回推理过程
                # 注意：LangChain的astream无法正确获取reasoning_content，需要直接使用OpenAI客户端
                import asyncio
                from openai import AsyncOpenAI
                import os
                
                # 创建OpenAI兼容的客户端（DeepSeek API兼容OpenAI格式）
                client = AsyncOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
                )
                
                full_answer = ""
                reasoning_content = ""
                answer_content = ""
                reasoning_chunk_count = 0
                answer_chunk_count = 0
                reasoning_buffer = ""
                answer_buffer = ""
                reasoning_buffer_size = 0
                answer_buffer_size = 0
                reasoning_done = False
                
                # 使用OpenAI客户端的流式API
                stream = await client.chat.completions.create(
                    model=state.get("model_name", "deepseek-reasoner"),
                    messages=answer_messages,
                    stream=True
                )
                
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # 提取推理内容（DeepSeek特有字段）
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_part = delta.reasoning_content
                        reasoning_content += reasoning_part
                        reasoning_buffer += reasoning_part
                        reasoning_buffer_size += len(reasoning_part)
                        reasoning_chunk_count += 1
                        
                        # 推理过程缓冲输出
                        if reasoning_buffer_size >= 20 or reasoning_chunk_count % 5 == 0:
                            yield {
                                "type": "reasoning_chunk",
                                "content": reasoning_buffer,
                                "done": False
                            }
                            reasoning_buffer = ""
                            reasoning_buffer_size = 0
                            await asyncio.sleep(0.01)
                    
                    # 提取答案内容
                    if hasattr(delta, 'content') and delta.content:
                        # 如果推理刚完成且还没标记，先标记推理完成
                        if reasoning_content and not reasoning_done:
                            # 发送剩余推理缓冲区内容
                            if reasoning_buffer:
                                yield {
                                    "type": "reasoning_chunk",
                                    "content": reasoning_buffer,
                                    "done": False
                                }
                                reasoning_buffer = ""
                                reasoning_buffer_size = 0
                            
                            # 标记推理完成
                            yield {
                                "type": "reasoning_chunk",
                                "content": "",
                                "done": True,
                                "full_reasoning": reasoning_content,
                                "need_requery": False,
                                "new_query": None,
                                "context_summary": "",
                                "requery_depth": 0
                            }
                            reasoning_done = True
                            logger.info(f"✅ DeepSeek Reasoner 推理完成: {len(reasoning_content)} 字符")
                        
                        answer_part = delta.content
                        answer_content += answer_part
                        answer_buffer += answer_part
                        answer_buffer_size += len(answer_part)
                        answer_chunk_count += 1
                        
                        # 答案缓冲输出
                        if answer_buffer_size >= 30 or answer_chunk_count % 3 == 0:
                            yield {
                                "type": "answer_chunk",
                                "content": answer_buffer,
                                "done": False
                            }
                            answer_buffer = ""
                            answer_buffer_size = 0
                            await asyncio.sleep(0.01)
                
                # 发送剩余内容
                if reasoning_buffer:
                    yield {
                        "type": "reasoning_chunk",
                        "content": reasoning_buffer,
                        "done": False
                    }
                
                if answer_buffer:
                    yield {
                        "type": "answer_chunk",
                        "content": answer_buffer,
                        "done": False
                    }
                
                # 如果有推理内容但还没标记完成，标记推理完成
                if reasoning_content and not reasoning_done:
                    yield {
                        "type": "reasoning_chunk",
                        "content": "",
                        "done": True,
                        "full_reasoning": reasoning_content,
                        "need_requery": False,
                        "new_query": None,
                        "context_summary": "",
                        "requery_depth": 0
                    }
                    logger.info(f"✅ DeepSeek Reasoner 推理完成: {len(reasoning_content)} 字符")
                
                # 答案生成完成
                logger.info(f"✅ DeepSeek Reasoner 答案生成完成: {len(answer_content)} 字符")
                
                yield {
                    "type": "answer_chunk",
                    "content": "",
                    "done": True,
                    "full_answer": answer_content,
                    "full_reasoning": reasoning_content
                }
                
                return  # 直接返回，不执行后续的深度思考逻辑
                
            else:
                logger.info("🧠 开始深度思考推理过程...")
                
                # 获取深度思考参数（默认只进行1次二次检索）
                max_requery_depth = state.get("max_requery_depth", 1)
                current_depth = 0
                reasoning_messages = []
                
                # 要求LLM看完整文档判断上下文是否充分
                reasoning_system_prompt = '''你是一个专业的保险文档问答助手。

请仔细阅读下面提供的文档内容，分析是否足以回答用户的问题。

**重要说明：**
1. **上下文展示策略**：为了提高分析效率，系统已经按照相关性排序，展示前20个最相关的实体、关系和文档片段。如果这些信息不足，你可以请求进行二次检索获取更多信息。

2. **⚠️ 极其重要：不同保险产品的条款差异**
   - 每个保险产品都有独立的保险条款，不同产品的条款内容可能完全不同
   - 例如：「传家宝」保险条款可能不对全残进行赔付，而「传家福」保险条款可能对全残进行赔付
   - **绝对不要**将一个产品的条款套用到另一个产品上
   - 如果当前文档块来自不同的保险产品，必须严格区分各自的条款内容
   - 如果上下文中没有找到用户询问的具体产品的信息，应该生成新的检索查询来获取正确的产品信息

3. **文档格式说明**：文档内容可能包含表格数据，其格式如下：
   - `[SOURCE:__TABLE_ENTITY_X__]` 表示该块对应的表格占位符，X为表格编号
   - `[CONTEXT]` 后面是与该表格相关的上下文文本，可能包含多个其他表格占位符，但请只关注当前SOURCE对应的表格
   - `[HTML_TABLE]` 后面是真实表格的HTML内容，包含了表格的详细数据

在分析时，请理解表格占位符与实际HTML表格的对应关系，从HTML表格中提取准确的数值、条款、费率等关键信息。

**任务流程：**

1. **理解问题**：说明你如何理解用户的问题（1-2句话）

2. **评估上下文**：判断当前文档和知识图谱信息是否包含足够的答案依据（2-3句话）

3. **决策**：
   - 如果上下文**充分**：说明你的推理逻辑（2-3句话），并在最后一行输出：`[CONTEXT_SUFFICIENT]`
   - 如果上下文**不足**：
     a) 先总结当前上下文中的关键信息（2-3句话）
     b) 说明还缺少什么关键信息（1-2句话）
     c) 在最后一行输出：`[CONTEXT_SUMMARY: 你的总结] | [NEW_QUERY: 你生成的新查询语句]`

**要求：**
- 使用第一人称（"我理解..."、"我发现..."）
- 仔细阅读所有文档内容，不要仅看预览
- 总字数200-400字
- 必须在最后一行明确输出决策标记'''
                
                reasoning_messages.append({"role": "system", "content": reasoning_system_prompt})
                
                # 添加历史对话
                if chat_history:
                    for msg in chat_history[-4:]:
                        reasoning_messages.append({
                            "role": msg.get("role"),
                            "content": msg.get("content")
                        })
                
                # 提供完整的上下文内容给LLM分析
                reasoning_user_message = f"""**用户问题：** {query}

{full_context}

请仔细阅读上述所有内容，分析上下文是否充分，并给出你的推理分析和决策。"""
                
                reasoning_messages.append({"role": "user", "content": reasoning_user_message})
                
                # 流式生成LLM推理部分
                import asyncio
                llm_reasoning = ""
                chunk_count = 0
                buffer = ""  # 缓冲区，累积小chunk
                buffer_size = 0
                
                async for chunk in llm.astream(reasoning_messages):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    llm_reasoning += content
                    buffer += content
                    buffer_size += len(content)
                    chunk_count += 1
                    
                    # 当缓冲区达到一定大小（例如20字符）或每5个chunk时才yield
                    if buffer_size >= 20 or chunk_count % 5 == 0:
                        yield {
                            "type": "reasoning_chunk",
                            "content": buffer,
                            "done": False
                        }
                        buffer = ""
                        buffer_size = 0
                        # 释放事件循环，避免阻塞
                        await asyncio.sleep(0.01)
                
                # 发送剩余缓冲区内容
                if buffer:
                    yield {
                        "type": "reasoning_chunk",
                        "content": buffer,
                        "done": False
                    }
                
                # 思考过程完成
                full_reasoning = llm_reasoning
                logger.info(f"✅ 思考推理完成: LLM推理 {len(llm_reasoning)} 字符")
                
                # === 判断是否需要二次检索 ===
                need_requery = False
                new_query = None
                context_summary = ""
                
                # 检查是否需要二次检索且未超过深度限制
                if "[CONTEXT_SUMMARY:" in llm_reasoning and "[NEW_QUERY:" in llm_reasoning:
                    if current_depth < max_requery_depth:
                        # 提取总结和新查询
                        import re
                        summary_match = re.search(r'\[CONTEXT_SUMMARY:\s*(.+?)\]\s*\|', llm_reasoning, re.DOTALL)
                        query_match = re.search(r'\[NEW_QUERY:\s*(.+?)\]', llm_reasoning)
                        
                        if summary_match and query_match:
                            context_summary = summary_match.group(1).strip()
                            new_query = query_match.group(1).strip()
                            need_requery = True
                            current_depth += 1
                            
                            logger.info(f"🔄 LLM判断需要二次检索 (深度 {current_depth}/{max_requery_depth})")
                            logger.info(f"📝 当前上下文总结: {context_summary[:100]}...")
                            logger.info(f"🔍 新查询: {new_query}")
                            
                            # 显示二次检索提示
                            requery_notice = f"\n\n---\n\n🔄 **深度思考：需要二次检索** (深度 {current_depth}/{max_requery_depth})\n\n**当前上下文总结：**\n{context_summary}\n\n**优化查询：** `{new_query}`\n\n正在进行二次检索...\n\n"
                            yield {
                                "type": "reasoning_chunk",
                                "content": requery_notice,
                                "done": False
                            }
                            
                            # === 执行二次检索 ===
                            logger.info("🔍 开始二次检索...")
                            try:
                                # 调用检索节点
                                requery_state = {
                                    "query": new_query,
                                    "query_mode": state.get("query_mode", "hybrid")
                                }
                                retrieval_result = await self.retrieve_context(requery_state)
                                
                                # 调用精排节点
                                rerank_state = {
                                    "query": new_query,
                                    "retrieved_docs": retrieval_result.get("retrieved_docs", []),
                                    "reranker": state.get("reranker"),
                                    "rerank_top_k": state.get("rerank_top_k")
                                }
                                rerank_result = await self.rerank_context(rerank_state)
                                
                                # 获取新上下文
                                new_final_docs = rerank_result.get("final_docs", [])
                                new_entities = retrieval_result.get("retrieved_entities", [])
                                new_relationships = retrieval_result.get("retrieved_relationships", [])
                                
                                # 构建新上下文
                                new_context = self._build_context(
                                    new_final_docs, new_entities, new_relationships
                                )
                                
                                logger.info(f"✅ 二次检索完成: {len(new_final_docs)} 个文档, {len(new_entities)} 个实体, {len(new_relationships)} 条关系")
                                
                                # 拼接总结与新上下文
                                full_context = f"""## 原始上下文总结

    {context_summary}

    ---

    ## 二次检索的新增上下文

    {new_context}"""
                                
                                # 更新状态变量
                                final_docs = new_final_docs
                                retrieved_entities = new_entities
                                retrieved_relationships = new_relationships
                                
                                # 显示二次检索结果
                                requery_result_notice = f"\n✅ **二次检索完成**\n- 新增文档: {len(new_final_docs)} 个\n- 新增实体: {len(new_entities)} 个\n- 新增关系: {len(new_relationships)} 条\n\n已将原始总结与新上下文拼接，正在生成增强回答...\n\n"
                                yield {
                                    "type": "reasoning_chunk",
                                    "content": requery_result_notice,
                                    "done": False
                                }
                                
                                # 更新reasoning记录
                                full_reasoning += requery_notice + requery_result_notice
                                
                            except Exception as e:
                                logger.error(f"❌ 二次检索失败: {e}")
                                error_notice = f"\n\n⚠️ 二次检索失败: {str(e)}\n将使用原始检索结果生成答案。\n\n"
                                yield {
                                    "type": "reasoning_chunk",
                                    "content": error_notice,
                                    "done": False
                                }
                                full_reasoning += error_notice
                    else:
                        logger.info(f"⚠️  已达最大检索深度 ({max_requery_depth})，不再进行二次检索")
                        depth_limit_notice = f"\n\n⚠️  已达最大深度思考次数 ({max_requery_depth})，将基于当前上下文生成答案。\n\n"
                        yield {
                            "type": "reasoning_chunk",
                            "content": depth_limit_notice,
                            "done": False
                        }
                        full_reasoning += depth_limit_notice
                
                # 标记推理完成
                yield {
                    "type": "reasoning_chunk",
                    "content": "",
                    "done": True,
                    "full_reasoning": full_reasoning,
                    "need_requery": need_requery,
                    "new_query": new_query,
                    "context_summary": context_summary,
                    "requery_depth": current_depth
                }
                
            # === 第二步：基于思考过程生成最终答案 ===
            logger.info("🤖 开始流式生成最终答案...")
            logger.info(f"📊 上下文统计:")
            logger.info(f"   - 实体: {len(retrieved_entities)} 个")
            logger.info(f"   - 关系: {len(retrieved_relationships)} 条")
            logger.info(f"   - 文档: {len(final_docs)} 个")
            
            # 重新构建完整上下文（不限制数量），供答案生成使用
            # 注意：如果进行了二次检索，full_context已经在上面被更新为包含新上下文的版本
            if not need_requery:
                # 如果没有进行二次检索，需要重新构建完整上下文（深度思考阶段只用了前20个）
                full_context = self._build_context(
                    final_docs, retrieved_entities, retrieved_relationships, max_items=None
                )
                logger.info("📝 已重新构建完整上下文（无限制）用于答案生成")
            
            answer_messages = []
            
            # 答案生成的系统提示
            answer_system_prompt = f'''你是一个专业的保险文档问答助手。
请根据下面提供的知识图谱信息和文档内容来回答用户的问题。

知识图谱包含了从文档中提取的实体和关系，提供了结构化的知识视图。
文档内容是经过精排的相关文本片段，按相关性从高到低排序。

**⚠️ 极其重要：不同保险产品的条款差异**
1. **产品独立性**：每个保险产品都有其独立的保险条款，不同产品的条款内容可能完全不同。
   - 例如：「传家宝」保险条款可能不对全残进行赔付
   - 而「传家福」保险条款可能对全残进行赔付
   
2. **严格匹配**：回答问题时，必须严格区分不同的保险产品：
   - 检查文档来源（file_path、chunk_id）确认产品名称
   - 不要将一个产品的条款套用到另一个产品上
   - 如果用户询问的是产品A，绝不能用产品B的条款来回答
   
3. **信息不足时的处理**：
   - 如果当前上下文中没有找到用户询问的具体产品的相关信息
   - 明确告知用户"当前检索的信息不包含该产品的相关条款"
   - 不要猜测或使用其他产品的信息进行推断

**文档格式说明**
文档内容可能包含表格数据，其格式如下：
- `[SOURCE:__TABLE_ENTITY_X__]` 表示该块对应的表格占位符，X为表格编号
- `[CONTEXT]` 后面是与该表格相关的上下文文本，可能包含多个其他表格占位符，但请只关注当前SOURCE对应的表格
- `[HTML_TABLE]` 后面是真实表格的HTML内容，包含了表格的详细数据

在回答时，请理解表格占位符与实际HTML表格的对应关系，从HTML表格中提取准确的数值、条款、费率等关键信息，确保数据的准确性。

**生成答案的要求：**
1. 优先利用知识图谱的结构化信息理解实体间的关系
2. 结合文档内容（包括表格数据）提供详细的上下文支持
3. 使用清晰、专业的语气
4. 如果可能，引用具体的实体、关系或文档来源
5. 如果信息不足，请直接告知
6. **重要**：你的分析思路已经在前面展示过了，请直接给出最终答案，不要重复思考过程的内容'''
            
            answer_messages.append({"role": "system", "content": answer_system_prompt})
            
            # 添加历史对话（最近5轮）
            if chat_history:
                for msg in chat_history[-10:]:
                    answer_messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content")
                    })
            
            # 添加刚才的思考过程作为上下文
            answer_messages.append({
                "role": "assistant",
                "content": f"【我的分析思路】\n{full_reasoning}"
            })
            
            # 添加当前查询（包含完整上下文）
            answer_user_message = f"""请基于以下保险知识库信息和你的分析思路，给出详细的答案:

{full_context}

**用户问题:** {query}"""
            
            answer_messages.append({"role": "user", "content": answer_user_message})
            
            # 流式生成最终答案
            full_answer = ""
            answer_buffer = ""
            answer_buffer_size = 0
            answer_chunk_count = 0
            
            async for chunk in llm.astream(answer_messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_answer += content
                answer_buffer += content
                answer_buffer_size += len(content)
                answer_chunk_count += 1
                
                # 当缓冲区达到一定大小（例如30字符）或每3个chunk时才yield
                if answer_buffer_size >= 30 or answer_chunk_count % 3 == 0:
                    yield {
                        "type": "answer_chunk",
                        "content": answer_buffer,
                        "done": False
                    }
                    answer_buffer = ""
                    answer_buffer_size = 0
                    # 释放事件循环，避免阻塞
                    await asyncio.sleep(0.01)
            
            # 发送剩余缓冲区内容
            if answer_buffer:
                yield {
                    "type": "answer_chunk",
                    "content": answer_buffer,
                    "done": False
                }
            
            # 标记完成
            logger.info(f"✅ 流式答案生成完成 (长度: {len(full_answer)} 字符)")
            yield {
                "type": "answer_chunk",
                "content": "",
                "done": True,
                "full_answer": full_answer,
                "full_reasoning": full_reasoning
            }
            
        except Exception as e:
            logger.error(f"❌ 流式答案生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield {
                "type": "answer_chunk",
                "content": f"生成答案时出错: {str(e)}",
                "done": True
            }
    
    def _build_context(self, final_docs, retrieved_entities, retrieved_relationships, max_items=None):
        """构建完整上下文字符串（供流式和非流式版本共用）
        
        Args:
            final_docs: 文档列表
            retrieved_entities: 实体列表
            retrieved_relationships: 关系列表
            max_items: 最大项目数量限制，None表示不限制。用于深度思考阶段减少信息量
        """
        kg_context_parts = []

        if retrieved_entities:
            entity_context = "### 相关实体\n\n"
            # 限制实体数量
            entities_to_show = retrieved_entities[:max_items] if max_items else retrieved_entities
            total_entities = len(retrieved_entities)
            
            for idx, entity in enumerate(entities_to_show, 1):
                entity_name = entity.get('entity_name', '未知')
                entity_type = entity.get('entity_type', '未知')
                description = entity.get('description', '无描述')
                entity_context += f"{idx}. **{entity_name}** ({entity_type})\n   {description}\n\n"
            
            # 如果有截断，提示用户
            if max_items and total_entities > max_items:
                entity_context += f"*（共{total_entities}个实体，此处仅展示前{max_items}个最相关的）*\n\n"
            
            kg_context_parts.append(entity_context)

        if retrieved_relationships:
            relation_context = "### 相关关系\n\n"
            # 限制关系数量
            relationships_to_show = retrieved_relationships[:max_items] if max_items else retrieved_relationships
            total_relationships = len(retrieved_relationships)
            
            for idx, rel in enumerate(relationships_to_show, 1):
                src = rel.get('src_id', '?')
                tgt = rel.get('tgt_id', '?')
                desc = rel.get('description', '无描述')
                weight = rel.get('weight', 0)
                relation_context += f"{idx}. {src} → {tgt} (权重: {weight:.2f})\n   {desc}\n\n"
            
            # 如果有截断，提示用户
            if max_items and total_relationships > max_items:
                relation_context += f"*（共{total_relationships}条关系，此处仅展示前{max_items}条最相关的）*\n\n"
            
            kg_context_parts.append(relation_context)

        doc_context_parts = []
        if final_docs:
            doc_context_parts.append("### 相关文档\n")
            # 限制文档数量
            docs_to_show = final_docs[:max_items] if max_items else final_docs
            total_docs = len(final_docs)
            
            for idx, doc in enumerate(docs_to_show, 1):
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
            
            # 如果有截断，提示用户
            if max_items and total_docs > max_items:
                doc_context_parts.append(f"\n*（共{total_docs}个文档，此处仅展示前{max_items}个最相关的）*\n")

        kg_context_str = "\n".join(kg_context_parts) if kg_context_parts else ""
        doc_context_str = "\n" + ("-" * 60 + "\n").join(doc_context_parts) if doc_context_parts else ""

        full_context = ""
        if kg_context_str:
            full_context += "## 知识图谱信息\n\n" + kg_context_str + "\n"
        if doc_context_str:
            full_context += "## 文档内容\n" + doc_context_str

        return full_context