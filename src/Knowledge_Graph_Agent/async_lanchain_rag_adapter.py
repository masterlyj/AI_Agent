import asyncio
import inspect
from asyncio import to_thread
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections.abc import AsyncIterator as _AsyncIterator
from langchain_core.messages import HumanMessage, SystemMessage

# 可自定义的重试异常集（根据需要扩展）
RETRY_EXCEPTIONS = (ConnectionError, TimeoutError)

def _is_async_callable(obj: Any, name: str = "invoke") -> bool:
    """检测 langchain_llm 是否有异步调用接口（比如 ainvoke / agenerate）"""
    return callable(getattr(obj, "ainvoke", None)) or callable(getattr(obj, "agenerate", None)) or inspect.iscoroutinefunction(getattr(obj, name, None))

def _get_sync_callable(obj: Any, name: str = "invoke") -> Callable:
    """优先返回 ainvoke/agenerate/ainvoke，否则返回 invoke"""
    if callable(getattr(obj, "ainvoke", None)):
        return getattr(obj, "ainvoke")
    if callable(getattr(obj, "agenerate", None)):
        return getattr(obj, "agenerate")
    return getattr(obj, name)

def _get_maybe_async_callable(obj: Any, name: str = "invoke") -> Callable:
    """返回一个统一的可 await 的调用函数：如果只有同步 invoke，则包装为 to_thread 使用"""
    # 优先异步 API
    if callable(getattr(obj, "ainvoke", None)):
        return getattr(obj, "ainvoke")
    if callable(getattr(obj, "agenerate", None)):
        return getattr(obj, "agenerate")
    # 回退到同步 invoke
    if callable(getattr(obj, name, None)):
        def sync_wrapper(*args, **kwargs):
            return getattr(obj, name)(*args, **kwargs)
        async def async_wrapper(*args, **kwargs):
            return await to_thread(sync_wrapper, *args, **kwargs)
        return async_wrapper
    raise AttributeError(f"LLM object has no callable {name} / a{name} / agenerate methods")

def _normalize_messages(prompt: str, system_prompt: Optional[str], history_messages: Optional[List[Dict[str, Any]]]):
    """把 lightRAG 风格的 prompt/system/history 转为 LangChain 消息列表"""
    msgs = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    if history_messages:
        for m in history_messages:
            role = m.get("role", "").lower()
            content = m.get("content", "")
            if role == "system":
                msgs.append(SystemMessage(content=content))
            else:
                msgs.append(HumanMessage(content=content) if role == "user" else HumanMessage(content=content))
    # 最后追加当前user prompt
    msgs.append(HumanMessage(content=prompt))
    return msgs

def create_lightrag_compatible_complete(langchain_llm, *, retry_attempts: int = 3, retry_min_wait: int = 4):
    """
    返回一个可供 LightRAG 调用的 async 完成函数，兼容流式/非流式与 COT。
    使用方法:
        llm_complete = create_lightrag_compatible_complete(my_langchain_llm)
        # 非流式
        result = await llm_complete(prompt, system_prompt="...")
        # 流式
        async for chunk in llm_complete(prompt, system_prompt="...", stream=True):
            ...
    """

    # 获取统一的可 await 的调用函数
    async_call = _get_maybe_async_callable(langchain_llm, name="invoke")

    @retry(
        stop=stop_after_attempt(retry_attempts),
        wait=wait_exponential(multiplier=1, min=retry_min_wait, max=60),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    )
    async def _invoke_llm(messages, stream: bool = False, **invoke_kwargs):
        """
        调用 LangChain LLM；若模型实现了流式返回，则直接返回 async iterable / iterator
        这里假设调用签名类似：llm.invoke(messages=..., stream=True/False, **kwargs)
        """
        # 过滤掉LightRAG传入的不兼容参数
        incompatible_params = [
            'hashing_kv', 'keyword_extraction', 'entity_spec', 'relation_spec',
            '_priority', 'global_config', 'query_param', 'enable_cot'
        ]
        filtered_kwargs = {k: v for k, v in invoke_kwargs.items() if k not in incompatible_params}
        
        # 多数 LangChain chat models 接受 messages 参数 -> 尝试统一接口
        try:
            result = await async_call(messages, stream=stream, **filtered_kwargs)
        except TypeError:
            # 部分 LangChain 版本/模型使用 (messages) positional
            result = await async_call(messages, **({"stream": stream} if stream else {}), **filtered_kwargs)
        return result

    async def llm_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        enable_cot: bool = False,
        token_tracker: Any = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        最终供 LightRAG 调用的函数。
        - 若 stream=False：返回 str
        - 若 stream=True：返回 AsyncIterator[str]（可用于 async for）
        """
        messages = _normalize_messages(prompt, system_prompt, history_messages)

        # 触发调用（可能是非流式或流式）
        result = await _invoke_llm(messages=messages, stream=stream, **kwargs)

        # 如果是可异步迭代的流式对象
        if stream and hasattr(result, "__aiter__"):
            async def stream_generator() -> AsyncIterator[str]:
                cot_active = False
                cot_started = False
                initial_content_seen = False
                final_usage = None

                try:
                    async for chunk in result:
                        # 支持不同返回结构：chunk 可能是 dict/obj 包含 choices/delta/content/reasoning_content/usage
                        # 我们做一些兼容性提取
                        # 1) 优先处理 streaming choices delta 结构（OpenAI style / vLLM style）
                        content = None
                        reasoning_content = None
                        # try common patterns
                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = getattr(chunk.choices[0], "delta", None)
                            if delta:
                                content = getattr(delta, "content", None)
                                reasoning_content = getattr(delta, "reasoning_content", None)
                        # fallback: chunk may be dict-like
                        if content is None and isinstance(chunk, dict):
                            # openai-like streaming chunk
                            choices = chunk.get("choices")
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                reasoning_content = delta.get("reasoning_content")
                        # fallback2: chunk might just be a string
                        if content is None and isinstance(chunk, str):
                            content = chunk

                        # usage extraction for final chunk
                        if hasattr(chunk, "usage"):
                            final_usage = getattr(chunk, "usage")

                        # COT streaming rules
                        if enable_cot:
                            if content:
                                if cot_active:
                                    # 结束 COT 流
                                    yield "</think>"
                                    cot_active = False
                                # 输出正常内容
                                initial_content_seen = True
                                yield content
                            elif reasoning_content:
                                # 只有 reasoning content
                                if not initial_content_seen and not cot_started:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True
                                if cot_active:
                                    yield reasoning_content
                            else:
                                # 跳过空 chunk
                                continue
                        else:
                            # 非 COT，只输出 content 字段或 chunk 本身
                            if content:
                                yield content
                            else:
                                # 若 chunk 是 dict 并包含 text 之类字段也尝试输出
                                if isinstance(chunk, dict) and "text" in chunk:
                                    yield chunk["text"]

                    # 流结束后如果 COT 仍然开启，关闭标签
                    if enable_cot and cot_active:
                        yield "</think>"
                        cot_active = False

                    # token_tracker add usage if possible
                    if token_tracker and final_usage:
                        try:
                            token_tracker.add_usage({
                                "prompt_tokens": getattr(final_usage, "prompt_tokens", final_usage.get("prompt_tokens", 0) if isinstance(final_usage, dict) else 0),
                                "completion_tokens": getattr(final_usage, "completion_tokens", final_usage.get("completion_tokens", 0) if isinstance(final_usage, dict) else 0),
                                "total_tokens": getattr(final_usage, "total_tokens", final_usage.get("total_tokens", 0) if isinstance(final_usage, dict) else 0),
                            })
                        except Exception:
                            # 忽略 token tracker 的任何异常以免影响流
                            pass

                finally:
                    # 若 result 提供 aclose()，尝试关闭（防止资源泄露）
                    aclose = getattr(result, "aclose", None)
                    if aclose and callable(aclose):
                        try:
                            await aclose()
                        except Exception:
                            pass

            return stream_generator()

        else:
            # 非流式：result 可能是一个包含 .content / .message / .choices 的对象，也可能直接是 str
            final_content = ""
            reasoning_content = ""
            usage = None

            # 尝试多种兼容取值方式
            if isinstance(result, str):
                final_content = result
            else:
                # LangChain 典型返回可能是一个 object 包含 .content 或 .generations 或 .choices
                final_content = getattr(result, "content", None) or getattr(result, "text", None) or ""
                # deepseek / vllm style
                reasoning_content = getattr(result, "reasoning_content", "") or ""
                # usage
                usage = getattr(result, "usage", None) or (getattr(result, "raw", None) and getattr(result.raw, "usage", None))

                # fallback when content is nested in choices/messages
                if not final_content:
                    # choices -> message -> content
                    choices = getattr(result, "choices", None)
                    if choices and len(choices) > 0:
                        message = getattr(choices[0], "message", None)
                        if message:
                            final_content = getattr(message, "content", "") or ""
                        else:
                            final_content = getattr(choices[0], "text", "") or ""

            # COT 非流式合并规则（与 LightRAG 保持一致）：
            # - 若 enable_cot 且 reasoning_content 非空且 final_content 为空：将 reasoning_content 包裹在 <think>..</think> 并作为输出前缀
            # - 若 enable_cot 且 both present：优先使用 final_content（与 LightRAG 规则一致）
            if enable_cot:
                if reasoning_content and (not final_content or final_content.strip() == ""):
                    # 只有 reasoning_content -> 包裹后输出
                    final_content = f"<think>{reasoning_content}</think>{final_content}"
                # else: 如果 both present，保留 final_content（忽略 reasoning_content）

            # token_tracker
            if token_tracker and usage:
                try:
                    token_tracker.add_usage({
                        "prompt_tokens": getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0),
                        "completion_tokens": getattr(usage, "completion_tokens", usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0),
                        "total_tokens": getattr(usage, "total_tokens", usage.get("total_tokens", 0) if isinstance(usage, dict) else 0),
                    })
                except Exception:
                    pass

            # 最后验证非空
            if not final_content or final_content.strip() == "":
                raise RuntimeError("LLM returned empty content")

            return final_content

    return llm_complete
