from __future__ import annotations

import httpx
from typing import Literal


class APIStatusError(Exception):
    """当API响应状态码为4xx或5xx时抛出"""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(
        self, message: str, *, response: httpx.Response, body: object | None
    ) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(Exception):
    def __init__(
        self, *, message: str = "连接错误。", request: httpx.Request
    ) -> None:
        super().__init__(message, request, body=None)


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400  # pyright: ignore[reportIncompatibleVariableOverride]


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401  # pyright: ignore[reportIncompatibleVariableOverride]


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="请求超时。", request=request)


class StorageNotInitializedError(RuntimeError):
    """当在初始化前尝试进行存储操作时抛出"""

    def __init__(self, storage_type: str = "存储"):
        super().__init__(
            f"{storage_type} 未初始化。请确保正确初始化：\n"
            f"\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()  # 必须\n"
            f"  \n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status()  # 流水线操作必须\n"

        )


class PipelineNotInitializedError(KeyError):
    """在流水线状态初始化前访问时抛出"""

    def __init__(self, namespace: str = ""):
        msg = (
            f"未找到流水线命名空间'{namespace}'。"
            f"这通常表示流水线状态未初始化。\n"
            f"\n"
            f"请在初始化存储后调用 'await initialize_pipeline_status()' ：\n"
            f"\n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status()\n"
            f"\n"
            f"完整初始化流程：\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()\n"
            f"  await initialize_pipeline_status()"
        )
        super().__init__(msg)
