from __future__ import annotations
import weakref

import asyncio
import html
import csv
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from hashlib import md5
from typing import Any, Protocol, Callable, TYPE_CHECKING, List, Optional
import numpy as np
from dotenv import load_dotenv

from .constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_FILE_PATH_LENGTH,
)

# 初始化logger的基本配置
logger = logging.getLogger("Knowledge_Graph_RAG")
logger.propagate = False  # 防止日志消息发送到根logger
logger.setLevel(logging.INFO)

# 如果没有handler，则添加控制台处理器
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 设置httpx库的日志级别为WARNING，防止其输出过多日志
logging.getLogger("httpx").setLevel(logging.WARNING)

# 全局导入pypinyin，并在启动时记录日志
try:
    import pypinyin

    _PYPINYIN_AVAILABLE = True
    # logger.info("pypinyin加载成功，可用于中文拼音排序")
except ImportError:
    pypinyin = None
    _PYPINYIN_AVAILABLE = False
    logger.warning(
        "未安装pypinyin，将使用简单字符串排序代替中文拼音排序。"
    )


async def safe_vdb_operation_with_exception(
    operation: Callable,
    operation_name: str,
    entity_name: str = "",
    max_retries: int = 3,
    retry_delay: float = 0.2,
    logger_func: Optional[Callable] = None,
) -> None:
    """
    Safely execute vector database operations with retry mechanism and exception handling.

    This function ensures that VDB operations are executed with proper error handling
    and retry logic. If all retries fail, it raises an exception to maintain data consistency.

    Args:
        operation: The async operation to execute
        operation_name: Operation name for logging purposes
        entity_name: Entity name for logging purposes
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        logger_func: Logger function to use for error messages

    Raises:
        Exception: When operation fails after all retry attempts
    """
    log_func = logger_func or logger.warning

    for attempt in range(max_retries):
        try:
            await operation()
            return  # Success, return immediately
        except Exception as e:
            if attempt >= max_retries - 1:
                error_msg = f"VDB {operation_name} failed for {entity_name} after {max_retries} attempts: {e}"
                log_func(error_msg)
                raise Exception(error_msg) from e
            else:
                log_func(
                    f"VDB {operation_name} attempt {attempt + 1} failed for {entity_name}: {e}, retrying..."
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)


def get_env_value(
    env_key: str, default: any, value_type: type = str, special_none: bool = False
) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to
        special_none (bool): If True, return None when value is "None"

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    # Handle special case for "None" string
    if special_none and value == "None":
        return None

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")

    # Handle list type with JSON parsing
    if value_type is list:
        try:
            import json

            parsed_value = json.loads(value)
            # Ensure the parsed value is actually a list
            if isinstance(parsed_value, list):
                return parsed_value
            else:
                logger.warning(
                    f"Environment variable {env_key} is not a valid JSON list, using default"
                )
                return default
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse {env_key} as JSON list: {e}, using default"
            )
            return default

    try:
        return value_type(value)
    except (ValueError, TypeError):
        return default


# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .base import BaseKVStorage, BaseVectorStorage, QueryParam

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"


def verbose_debug(msg: str, *args, **kwargs):
    """Function for outputting detailed debug information.
    When VERBOSE_DEBUG=True, outputs the complete message.
    When VERBOSE_DEBUG=False, outputs only the first 50 characters.

    Args:
        msg: The message format string
        *args: Arguments to be formatted into the message
        **kwargs: Keyword arguments passed to logger.debug()
    """
    if VERBOSE_DEBUG:
        logger.debug(msg, *args, **kwargs)
    else:
        # Format the message with args first
        if args:
            formatted_msg = msg % args
        else:
            formatted_msg = msg
        # Then truncate the formatted message
        truncated_msg = (
            formatted_msg[:150] + "..." if len(formatted_msg) > 150 else formatted_msg
        )
        # Remove consecutive newlines
        truncated_msg = re.sub(r"\n+", "\n", truncated_msg)
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    """Enable or disable verbose debug output"""
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""

    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = [
            "/documents",
            "/documents/paginated",
            "/health",
            "/webui/",
            "/documents/pipeline_status",
        ]
        # self.filtered_paths = ["/health", "/webui/"]

    def filter(self, record):
        try:
            # Check if record has the required attributes for an access log
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            # Extract method, path and status from the record args
            method = record.args[1]
            path = record.args[2]
            status = record.args[4]

            # Filter out successful GET/POST requests to filtered paths
            if (
                (method == "GET" or method == "POST")
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True
        except Exception:
            # In case of any error, let the message through
            return True


def setup_logger(
    logger_name: str,
    level: str = "INFO",
    add_filter: bool = False,
    log_file_path: str | None = None,
    enable_file_logging: bool = True,
):
    """
    配置日志记录器，支持控制台和可选的文件输出。

    Args:
        logger_name: 日志记录器的名称
        level: 日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL）
        add_filter: 是否添加 LightragPathFilter 过滤器
        log_file_path: 日志文件路径。如果为 None 且启用文件日志，则默认在 LOG_DIR 或当前目录下创建 lightrag.log
        enable_file_logging: 是否启用文件日志（默认启用）
    """
    # 配置日志格式
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(level)
    logger_instance.handlers = []  # 清除已有处理器
    logger_instance.propagate = False

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    logger_instance.addHandler(console_handler)

    # 默认启用文件处理器，除非显式禁用
    if enable_file_logging:
        # 获取日志文件路径
        if log_file_path is None:
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # 从环境变量读取日志文件大小上限和备份数量
        log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
        log_backup_count = get_env_value(
            "LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int
        )

        try:
            # 添加文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(level)
            logger_instance.addHandler(file_handler)
        except PermissionError as e:
            logger.warning(f"无法在 {log_file_path} 创建日志文件：{str(e)}")
            logger.warning("仅使用控制台日志继续运行")

    # 如果需要，添加路径过滤器
    if add_filter:
        path_filter = LightragPathFilter()
        logger_instance.addFilter(path_filter)


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class TaskState:
    """Task state tracking for priority queue management"""

    future: asyncio.Future
    start_time: float
    execution_start_time: float = None
    worker_started: bool = False
    cancellation_requested: bool = False
    cleanup_done: bool = False


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    func: callable
    max_token_size: int | None = None  # deprecated keep it for compatible only

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling.

    Args:
        *args: Arguments to hash
    Returns:
        str: Hash string
    """
    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])

    # Use 'replace' error handling to safely encode problematic Unicode characters
    # This replaces invalid characters with Unicode replacement character (U+FFFD)
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key in the format {mode}:{cache_type}:{hash}

    Args:
        mode: Cache mode (e.g., 'default', 'local', 'global')
        cache_type: Type of cache (e.g., 'extract', 'query', 'keywords')
        hash_value: Hash value from compute_args_hash

    Returns:
        str: Flattened cache key
    """
    return f"{mode}:{cache_type}:{hash_value}"


def parse_cache_key(cache_key: str) -> tuple[str, str, str] | None:
    """Parse a flattened cache key back into its components

    Args:
        cache_key: Flattened cache key in format {mode}:{cache_type}:{hash}

    Returns:
        tuple[str, str, str] | None: (mode, cache_type, hash) or None if invalid format
    """
    parts = cache_key.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


# Custom exception classes
class QueueFullError(Exception):
    """Raised when the queue is full and the wait times out"""

    pass


class WorkerTimeoutError(Exception):
    """Worker-level timeout exception with specific timeout information"""

    def __init__(self, timeout_value: float, timeout_type: str = "execution"):
        self.timeout_value = timeout_value
        self.timeout_type = timeout_type
        super().__init__(f"Worker {timeout_type} timeout after {timeout_value}s")


class HealthCheckTimeoutError(Exception):
    """Health Check-level timeout exception"""

    def __init__(self, timeout_value: float, execution_duration: float):
        self.timeout_value = timeout_value
        self.execution_duration = execution_duration
        super().__init__(
            f"Task forcefully terminated due to execution timeout (>{timeout_value}s, actual: {execution_duration:.1f}s)"
        )


def priority_limit_async_func_call(
    max_size: int,
    llm_timeout: float = None,
    max_execution_timeout: float = None,
    max_task_duration: float = None,
    max_queue_size: int = 1000,
    cleanup_timeout: float = 2.0,
    queue_name: str = "limit_async",
):
    """
    具有强大超时处理功能的增强型优先级受限异步函数调用装饰器

    此装饰器为管理并发的大语言模型（LLM）请求提供了全面的解决方案，包括：
    - 多层超时保护（大语言模型 -> 工作线程 -> 健康检查 -> 用户）
    - 任务状态跟踪，以防止竞态条件
    - 增强的健康检查系统，具备卡住任务检测功能
    - 适当的资源清理和错误恢复机制

    参数：
        max_size：最大并发调用数量
        max_queue_size：最大队列容量，以防止内存溢出
        llm_timeout：大语言模型提供商超时时间（来自全局配置），用于计算其他超时时间
        max_execution_timeout：工作线程执行函数的最长时间（默认为llm_timeout + 30秒）
        max_task_duration：健康检查介入前的最长时间（默认为llm_timeout + 60秒）
        cleanup_timeout：等待清理操作的最长时间（默认为2.0秒）
        queue_name：用于日志识别的可选队列名称（默认为“limit_async”）

    返回：
        装饰器函数
    """

    def final_decro(func):
        # Ensure func is callable
        if not callable(func):
            raise TypeError(f"Expected a callable object, got {type(func)}")

        # Calculate timeout hierarchy if llm_timeout is provided (Dynamic Timeout Calculation)
        if llm_timeout is not None:
            nonlocal max_execution_timeout, max_task_duration
            if max_execution_timeout is None:
                max_execution_timeout = (
                    llm_timeout * 2
                )  # Reserved timeout buffer for low-level retry
            if max_task_duration is None:
                max_task_duration = (
                    llm_timeout * 2 + 15
                )  # Reserved timeout buffer for health check phase

        queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        tasks = set()
        initialization_lock = asyncio.Lock()
        counter = 0
        shutdown_event = asyncio.Event()
        initialized = False
        worker_health_check_task = None

        # Enhanced task state management
        task_states = {}  # task_id -> TaskState
        task_states_lock = asyncio.Lock()
        active_futures = weakref.WeakSet()
        reinit_count = 0

        async def worker():
            """Enhanced worker that processes tasks with proper timeout and state management"""
            try:
                while not shutdown_event.is_set():
                    try:
                        # Get task from queue with timeout for shutdown checking
                        try:
                            (
                                priority,
                                count,
                                task_id,
                                args,
                                kwargs,
                            ) = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        # Get task state and mark worker as started
                        async with task_states_lock:
                            if task_id not in task_states:
                                queue.task_done()
                                continue
                            task_state = task_states[task_id]
                            task_state.worker_started = True
                            # Record execution start time when worker actually begins processing
                            task_state.execution_start_time = (
                                asyncio.get_event_loop().time()
                            )

                        # Check if task was cancelled before worker started
                        if (
                            task_state.cancellation_requested
                            or task_state.future.cancelled()
                        ):
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()
                            continue

                        try:
                            # Execute function with timeout protection
                            if max_execution_timeout is not None:
                                result = await asyncio.wait_for(
                                    func(*args, **kwargs), timeout=max_execution_timeout
                                )
                            else:
                                result = await func(*args, **kwargs)

                            # Set result if future is still valid
                            if not task_state.future.done():
                                task_state.future.set_result(result)

                        except asyncio.TimeoutError:
                            # Worker-level timeout (max_execution_timeout exceeded)
                            logger.warning(
                                f"{queue_name}: Worker timeout for task {task_id} after {max_execution_timeout}s"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(
                                    WorkerTimeoutError(
                                        max_execution_timeout, "execution"
                                    )
                                )
                        except asyncio.CancelledError:
                            # Task was cancelled during execution
                            if not task_state.future.done():
                                task_state.future.cancel()
                            logger.debug(
                                f"{queue_name}: Task {task_id} cancelled during execution"
                            )
                        except Exception as e:
                            # Function execution error
                            logger.error(
                                f"{queue_name}: Error in decorated function for task {task_id}: {str(e)}"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(e)
                        finally:
                            # Clean up task state
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()

                    except Exception as e:
                        # Critical error in worker loop
                        logger.error(
                            f"{queue_name}: Critical error in worker: {str(e)}"
                        )
                        await asyncio.sleep(0.1)
            finally:
                logger.debug(f"{queue_name}: Worker exiting")

        async def enhanced_health_check():
            """Enhanced health check with stuck task detection and recovery"""
            nonlocal initialized
            try:
                while not shutdown_event.is_set():
                    await asyncio.sleep(5)  # Check every 5 seconds

                    current_time = asyncio.get_event_loop().time()

                    # Detect and handle stuck tasks based on execution start time
                    if max_task_duration is not None:
                        stuck_tasks = []
                        async with task_states_lock:
                            for task_id, task_state in list(task_states.items()):
                                # Only check tasks that have started execution
                                if (
                                    task_state.worker_started
                                    and task_state.execution_start_time is not None
                                    and current_time - task_state.execution_start_time
                                    > max_task_duration
                                ):
                                    stuck_tasks.append(
                                        (
                                            task_id,
                                            current_time
                                            - task_state.execution_start_time,
                                        )
                                    )

                        # Force cleanup of stuck tasks
                        for task_id, execution_duration in stuck_tasks:
                            logger.warning(
                                f"{queue_name}: Detected stuck task {task_id} (execution time: {execution_duration:.1f}s), forcing cleanup"
                            )
                            async with task_states_lock:
                                if task_id in task_states:
                                    task_state = task_states[task_id]
                                    if not task_state.future.done():
                                        task_state.future.set_exception(
                                            HealthCheckTimeoutError(
                                                max_task_duration, execution_duration
                                            )
                                        )
                                    task_states.pop(task_id, None)

                    # Worker recovery logic
                    current_tasks = set(tasks)
                    done_tasks = {t for t in current_tasks if t.done()}
                    tasks.difference_update(done_tasks)

                    active_tasks_count = len(tasks)
                    workers_needed = max_size - active_tasks_count

                    if workers_needed > 0:
                        logger.info(
                            f"{queue_name}: Creating {workers_needed} new workers"
                        )
                        new_tasks = set()
                        for _ in range(workers_needed):
                            task = asyncio.create_task(worker())
                            new_tasks.add(task)
                            task.add_done_callback(tasks.discard)
                        tasks.update(new_tasks)

            except Exception as e:
                logger.error(f"{queue_name}: Error in enhanced health check: {str(e)}")
            finally:
                logger.debug(f"{queue_name}: Enhanced health check task exiting")
                initialized = False

        async def ensure_workers():
            """Ensure worker system is initialized with enhanced error handling"""
            nonlocal initialized, worker_health_check_task, tasks, reinit_count

            if initialized:
                return

            async with initialization_lock:
                if initialized:
                    return

                if reinit_count > 0:
                    reinit_count += 1
                    logger.warning(
                        f"{queue_name}: Reinitializing system (count: {reinit_count})"
                    )
                else:
                    reinit_count = 1

                # Clean up completed tasks
                current_tasks = set(tasks)
                done_tasks = {t for t in current_tasks if t.done()}
                tasks.difference_update(done_tasks)

                active_tasks_count = len(tasks)
                if active_tasks_count > 0 and reinit_count > 1:
                    logger.warning(
                        f"{queue_name}: {active_tasks_count} tasks still running during reinitialization"
                    )

                # Create worker tasks
                workers_needed = max_size - active_tasks_count
                for _ in range(workers_needed):
                    task = asyncio.create_task(worker())
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                # Start enhanced health check
                worker_health_check_task = asyncio.create_task(enhanced_health_check())

                initialized = True
                # Log dynamic timeout configuration
                timeout_info = []
                if llm_timeout is not None:
                    timeout_info.append(f"Func: {llm_timeout}s")
                if max_execution_timeout is not None:
                    timeout_info.append(f"Worker: {max_execution_timeout}s")
                if max_task_duration is not None:
                    timeout_info.append(f"Health Check: {max_task_duration}s")

                timeout_str = (
                    f"(Timeouts: {', '.join(timeout_info)})" if timeout_info else ""
                )
                logger.info(
                    f"{queue_name}: {workers_needed} new workers initialized {timeout_str}"
                )

        async def shutdown():
            """Gracefully shut down all workers and cleanup resources"""
            logger.info(f"{queue_name}: Shutting down priority queue workers")

            shutdown_event.set()

            # Cancel all active futures
            for future in list(active_futures):
                if not future.done():
                    future.cancel()

            # Cancel all pending tasks
            async with task_states_lock:
                for task_id, task_state in list(task_states.items()):
                    if not task_state.future.done():
                        task_state.future.cancel()
                task_states.clear()

            # Wait for queue to empty with timeout
            try:
                await asyncio.wait_for(queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"{queue_name}: Timeout waiting for queue to empty during shutdown"
                )

            # Cancel worker tasks
            for task in list(tasks):
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Cancel health check task
            if worker_health_check_task and not worker_health_check_task.done():
                worker_health_check_task.cancel()
                try:
                    await worker_health_check_task
                except asyncio.CancelledError:
                    pass

            logger.info(f"{queue_name}: Priority queue workers shutdown complete")

        @wraps(func)
        async def wait_func(
            *args, _priority=10, _timeout=None, _queue_timeout=None, **kwargs
        ):
            """
            Execute function with enhanced priority-based concurrency control and timeout handling

            Args:
                *args: Positional arguments passed to the function
                _priority: Call priority (lower values have higher priority)
                _timeout: Maximum time to wait for completion (in seconds, none means determinded by max_execution_timeout of the queue)
                _queue_timeout: Maximum time to wait for entering the queue (in seconds)
                **kwargs: Keyword arguments passed to the function

            Returns:
                The result of the function call

            Raises:
                TimeoutError: If the function call times out at any level
                QueueFullError: If the queue is full and waiting times out
                Any exception raised by the decorated function
            """
            await ensure_workers()

            # Generate unique task ID
            task_id = f"{id(asyncio.current_task())}_{asyncio.get_event_loop().time()}"
            future = asyncio.Future()

            # Create task state
            task_state = TaskState(
                future=future, start_time=asyncio.get_event_loop().time()
            )

            try:
                # Register task state
                async with task_states_lock:
                    task_states[task_id] = task_state

                active_futures.add(future)

                # Get counter for FIFO ordering
                nonlocal counter
                async with initialization_lock:
                    current_count = counter
                    counter += 1

                # Queue the task with timeout handling
                try:
                    if _queue_timeout is not None:
                        await asyncio.wait_for(
                            queue.put(
                                (_priority, current_count, task_id, args, kwargs)
                            ),
                            timeout=_queue_timeout,
                        )
                    else:
                        await queue.put(
                            (_priority, current_count, task_id, args, kwargs)
                        )
                except asyncio.TimeoutError:
                    raise QueueFullError(
                        f"{queue_name}: Queue full, timeout after {_queue_timeout} seconds"
                    )
                except Exception as e:
                    # Clean up on queue error
                    if not future.done():
                        future.set_exception(e)
                    raise

                # Wait for result with timeout handling
                try:
                    if _timeout is not None:
                        return await asyncio.wait_for(future, _timeout)
                    else:
                        return await future
                except asyncio.TimeoutError:
                    # This is user-level timeout (asyncio.wait_for caused)
                    # Mark cancellation request
                    async with task_states_lock:
                        if task_id in task_states:
                            task_states[task_id].cancellation_requested = True

                    # Cancel future
                    if not future.done():
                        future.cancel()

                    # Wait for worker cleanup with timeout
                    cleanup_start = asyncio.get_event_loop().time()
                    while (
                        task_id in task_states
                        and asyncio.get_event_loop().time() - cleanup_start
                        < cleanup_timeout
                    ):
                        await asyncio.sleep(0.1)

                    raise TimeoutError(
                        f"{queue_name}: User timeout after {_timeout} seconds"
                    )
                except WorkerTimeoutError as e:
                    # This is Worker-level timeout, directly propagate exception information
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except HealthCheckTimeoutError as e:
                    # This is Health Check-level timeout, directly propagate exception information
                    raise TimeoutError(f"{queue_name}: {str(e)}")

            finally:
                # Ensure cleanup
                active_futures.discard(future)
                async with task_states_lock:
                    task_states.pop(task_id, None)

        # Add shutdown method to decorated function
        wait_func.shutdown = shutdown

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """
    封装分词器，提供统一的编码（encode）和解码（decode）接口。
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        初始化Tokenizer，指定模型名称及实际分词器实例。

        参数:
            model_name: 分词器关联的模型名称。
            tokenizer: 实现了TokenizerInterface的分词器对象实例。
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        使用底层分词器将字符串编码为token列表。

        参数:
            content: 待编码的字符串。

        返回:
            token整型列表。
        """
        return self.tokenizer.encode(content)

    def decode(self, tokens: List[int]) -> str:
        """
        使用底层分词器将token列表解码为字符串。

        参数:
            tokens: 需要解码的整数token列表。

        返回:
            解码后的字符串。
        """
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """
    A Tokenizer implementation using the tiktoken library.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the TiktokenTokenizer with a specified model name.

        Args:
            model_name: The model name for the tiktoken tokenizer to use.  Defaults to "gpt-4o-mini".

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If the model_name is invalid.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` or define custom `tokenizer_func`."
            )

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
) -> tuple[str, int] | None:
    """Generic cache handling function with flattened cache keys

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save
    """
    # Skip if storage is None or content is a streaming response
    if hashing_kv is None or not cache_data.content:
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    # Check if we already have identical content cached
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    # Create cache entry with flattened structure
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")

    # Save using flattened key
    await hashing_kv.upsert({flattened_key: cache_entry})


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def exists_func(obj, func_name: str) -> bool:
    """Check if a function exists in an object or not.
    :param obj:
    :param func_name:
    :return: True / False
    """
    if callable(getattr(obj, func_name, None)):
        return True
    else:
        return False


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


async def aexport_data(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    output_path: str,
    file_format: str = "csv",
    include_vector_data: bool = False,
) -> None:
    """
    Asynchronously exports all entities, relations, and relationships to various formats.

    Args:
        chunk_entity_relation_graph: Graph storage instance for entities and relations
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        output_path: The path to the output file (including extension).
        file_format: Output format - "csv", "excel", "md", "txt".
            - csv: Comma-separated values file
            - excel: Microsoft Excel file with multiple sheets
            - md: Markdown tables
            - txt: Plain text formatted output
        include_vector_data: Whether to include data from the vector database.
    """
    # Collect data
    entities_data = []
    relations_data = []
    relationships_data = []

    # --- Entities ---
    all_entities = await chunk_entity_relation_graph.get_all_labels()
    for entity_name in all_entities:
        # Get entity information from graph
        node_data = await chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        entity_info = {
            "graph_data": node_data,
            "source_id": source_id,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = await entities_vdb.get_by_id(entity_id)
            entity_info["vector_data"] = vector_data

        entity_row = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": str(
                entity_info["graph_data"]
            ),  # Convert to string to ensure compatibility
        }
        if include_vector_data and "vector_data" in entity_info:
            entity_row["vector_data"] = str(entity_info["vector_data"])
        entities_data.append(entity_row)

    # --- Relations ---
    for src_entity in all_entities:
        for tgt_entity in all_entities:
            if src_entity == tgt_entity:
                continue

            edge_exists = await chunk_entity_relation_graph.has_edge(
                src_entity, tgt_entity
            )
            if edge_exists:
                # Get edge information from graph
                edge_data = await chunk_entity_relation_graph.get_edge(
                    src_entity, tgt_entity
                )
                source_id = edge_data.get("source_id") if edge_data else None

                relation_info = {
                    "graph_data": edge_data,
                    "source_id": source_id,
                }

                # Optional: Get vector database information
                if include_vector_data:
                    rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
                    vector_data = await relationships_vdb.get_by_id(rel_id)
                    relation_info["vector_data"] = vector_data

                relation_row = {
                    "src_entity": src_entity,
                    "tgt_entity": tgt_entity,
                    "source_id": relation_info["source_id"],
                    "graph_data": str(relation_info["graph_data"]),  # Convert to string
                }
                if include_vector_data and "vector_data" in relation_info:
                    relation_row["vector_data"] = str(relation_info["vector_data"])
                relations_data.append(relation_row)

    # --- Relationships (from VectorDB) ---
    all_relationships = await relationships_vdb.client_storage
    for rel in all_relationships["data"]:
        relationships_data.append(
            {
                "relationship_id": rel["__id__"],
                "data": str(rel),  # Convert to string for compatibility
            }
        )

    # Export based on format
    if file_format == "csv":
        # CSV export
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            # Entities
            if entities_data:
                csvfile.write("# ENTITIES\n")
                writer = csv.DictWriter(csvfile, fieldnames=entities_data[0].keys())
                writer.writeheader()
                writer.writerows(entities_data)
                csvfile.write("\n\n")

            # Relations
            if relations_data:
                csvfile.write("# RELATIONS\n")
                writer = csv.DictWriter(csvfile, fieldnames=relations_data[0].keys())
                writer.writeheader()
                writer.writerows(relations_data)
                csvfile.write("\n\n")

            # Relationships
            if relationships_data:
                csvfile.write("# RELATIONSHIPS\n")
                writer = csv.DictWriter(
                    csvfile, fieldnames=relationships_data[0].keys()
                )
                writer.writeheader()
                writer.writerows(relationships_data)

    elif file_format == "excel":
        # Excel export
        import pandas as pd

        entities_df = pd.DataFrame(entities_data) if entities_data else pd.DataFrame()
        relations_df = (
            pd.DataFrame(relations_data) if relations_data else pd.DataFrame()
        )
        relationships_df = (
            pd.DataFrame(relationships_data) if relationships_data else pd.DataFrame()
        )

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            if not entities_df.empty:
                entities_df.to_excel(writer, sheet_name="Entities", index=False)
            if not relations_df.empty:
                relations_df.to_excel(writer, sheet_name="Relations", index=False)
            if not relationships_df.empty:
                relationships_df.to_excel(
                    writer, sheet_name="Relationships", index=False
                )

    elif file_format == "md":
        # Markdown export
        with open(output_path, "w", encoding="utf-8") as mdfile:
            mdfile.write("# LightRAG Data Export\n\n")

            # Entities
            mdfile.write("## Entities\n\n")
            if entities_data:
                # Write header
                mdfile.write("| " + " | ".join(entities_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(entities_data[0].keys())) + " |\n"
                )

                # Write rows
                for entity in entities_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in entity.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No entity data available*\n\n")

            # Relations
            mdfile.write("## Relations\n\n")
            if relations_data:
                # Write header
                mdfile.write("| " + " | ".join(relations_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(relations_data[0].keys())) + " |\n"
                )

                # Write rows
                for relation in relations_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in relation.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No relation data available*\n\n")

            # Relationships
            mdfile.write("## Relationships\n\n")
            if relationships_data:
                # Write header
                mdfile.write("| " + " | ".join(relationships_data[0].keys()) + " |\n")
                mdfile.write(
                    "| "
                    + " | ".join(["---"] * len(relationships_data[0].keys()))
                    + " |\n"
                )

                # Write rows
                for relationship in relationships_data:
                    mdfile.write(
                        "| "
                        + " | ".join(str(v) for v in relationship.values())
                        + " |\n"
                    )
            else:
                mdfile.write("*No relationship data available*\n\n")

    elif file_format == "txt":
        # Plain text export
        with open(output_path, "w", encoding="utf-8") as txtfile:
            txtfile.write("LIGHTRAG DATA EXPORT\n")
            txtfile.write("=" * 80 + "\n\n")

            # Entities
            txtfile.write("ENTITIES\n")
            txtfile.write("-" * 80 + "\n")
            if entities_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(e[k])) for e in entities_data))
                    for k in entities_data[0]
                }
                header = "  ".join(k.ljust(col_widths[k]) for k in entities_data[0])
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for entity in entities_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in entity.items()
                    )
                    txtfile.write(row + "\n")
                txtfile.write("\n\n")
            else:
                txtfile.write("No entity data available\n\n")

            # Relations
            txtfile.write("RELATIONS\n")
            txtfile.write("-" * 80 + "\n")
            if relations_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(r[k])) for r in relations_data))
                    for k in relations_data[0]
                }
                header = "  ".join(k.ljust(col_widths[k]) for k in relations_data[0])
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for relation in relations_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in relation.items()
                    )
                    txtfile.write(row + "\n")
                txtfile.write("\n\n")
            else:
                txtfile.write("No relation data available\n\n")

            # Relationships
            txtfile.write("RELATIONSHIPS\n")
            txtfile.write("-" * 80 + "\n")
            if relationships_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(r[k])) for r in relationships_data))
                    for k in relationships_data[0]
                }
                header = "  ".join(
                    k.ljust(col_widths[k]) for k in relationships_data[0]
                )
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for relationship in relationships_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in relationship.items()
                    )
                    txtfile.write(row + "\n")
            else:
                txtfile.write("No relationship data available\n\n")

    else:
        raise ValueError(
            f"Unsupported file format: {file_format}. "
            f"Choose from: csv, excel, md, txt"
        )
    if file_format is not None:
        print(f"Data exported to: {output_path} with format: {file_format}")
    else:
        print("Data displayed as table format")


def export_data(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    output_path: str,
    file_format: str = "csv",
    include_vector_data: bool = False,
) -> None:
    """
    Synchronously exports all entities, relations, and relationships to various formats.

    Args:
        chunk_entity_relation_graph: Graph storage instance for entities and relations
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        output_path: The path to the output file (including extension).
        file_format: Output format - "csv", "excel", "md", "txt".
            - csv: Comma-separated values file
            - excel: Microsoft Excel file with multiple sheets
            - md: Markdown tables
            - txt: Plain text formatted output
        include_vector_data: Whether to include data from the vector database.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        aexport_data(
            chunk_entity_relation_graph,
            entities_vdb,
            relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )
    )


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


async def update_chunk_cache_list(
    chunk_id: str,
    text_chunks_storage: "BaseKVStorage",
    cache_keys: list[str],
    cache_scenario: str = "batch_update",
) -> None:
    """Update chunk's llm_cache_list with the given cache keys

    Args:
        chunk_id: Chunk identifier
        text_chunks_storage: Text chunks storage instance
        cache_keys: List of cache keys to add to the list
        cache_scenario: Description of the cache scenario for logging
    """
    if not cache_keys:
        return

    try:
        chunk_data = await text_chunks_storage.get_by_id(chunk_id)
        if chunk_data:
            # Ensure llm_cache_list exists
            if "llm_cache_list" not in chunk_data:
                chunk_data["llm_cache_list"] = []

            # Add cache keys to the list if not already present
            existing_keys = set(chunk_data["llm_cache_list"])
            new_keys = [key for key in cache_keys if key not in existing_keys]

            if new_keys:
                chunk_data["llm_cache_list"].extend(new_keys)

                # Update the chunk in storage
                await text_chunks_storage.upsert({chunk_id: chunk_data})
                logger.debug(
                    f"Updated chunk {chunk_id} with {len(new_keys)} cache keys ({cache_scenario})"
                )
    except Exception as e:
        logger.warning(
            f"Failed to update chunk {chunk_id} with cache references on {cache_scenario}: {e}"
        )


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from the text
    Remove  orphon ...</think> tags from the text also"""
    return re.sub(
        r"^(<think>.*?</think>|.*</think>)", "", text, flags=re.DOTALL
    ).strip()


async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
) -> tuple[str, int]:
    """Call LLM function with cache support and text sanitization

    If cache is available and enabled (determined by handle_cache based on mode),
    retrieve result from cache; otherwise call LLM function and save result to cache.

    This function applies text sanitization to prevent UTF-8 encoding errors for all LLM providers.

    Args:
        input_text: Input text to send to LLM
        use_llm_func: LLM function with higher priority
        llm_response_cache: Cache storage instance
        max_tokens: Maximum tokens for generation
        history_messages: History messages list
        cache_type: Type of cache
        chunk_id: Chunk identifier to store in cache
        text_chunks_storage: Text chunks storage to update llm_cache_list
        cache_keys_collector: Optional list to collect cache keys for batch processing

    Returns:
        tuple[str, int]: (LLM response text, timestamp)
            - For cache hits: (content, cache_create_time)
            - For cache misses: (content, current_timestamp)
    """
    # Sanitize input text to prevent UTF-8 encoding errors for all LLM providers
    safe_user_prompt = sanitize_text_for_encoding(user_prompt)
    safe_system_prompt = (
        sanitize_text_for_encoding(system_prompt) if system_prompt else None
    )

    # Sanitize history messages if provided
    safe_history_messages = None
    if history_messages:
        safe_history_messages = []
        for i, msg in enumerate(history_messages):
            safe_msg = msg.copy()
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text_for_encoding(safe_msg["content"])
            safe_history_messages.append(safe_msg)
        history = json.dumps(safe_history_messages, ensure_ascii=False)
    else:
        history = None

    if llm_response_cache:
        prompt_parts = []
        if safe_user_prompt:
            prompt_parts.append(safe_user_prompt)
        if safe_system_prompt:
            prompt_parts.append(safe_system_prompt)
        if history:
            prompt_parts.append(history)
        _prompt = "\n".join(prompt_parts)

        arg_hash = compute_args_hash(_prompt)
        # Generate cache key for this LLM call
        cache_key = generate_cache_key("default", cache_type, arg_hash)

        cached_result = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
        )
        if cached_result:
            content, timestamp = cached_result
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

            return content, timestamp
        statistic_data["llm_call"] += 1

        # Call LLM with sanitized input
        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )

        res = remove_think_tags(res)

        # Generate timestamp for cache miss (LLM call completion time)
        current_timestamp = int(time.time())

        if llm_response_cache.global_config.get("enable_llm_cache_for_entity_extract"):
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type=cache_type,
                    chunk_id=chunk_id,
                ),
            )

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

        return res, current_timestamp

    # When cache is disabled, directly call LLM with sanitized input
    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
    except Exception as e:
        # Add [LLM func] prefix to error message
        error_msg = f"[LLM func] {str(e)}"
        # Re-raise with the same exception type but modified message
        raise type(e)(error_msg) from e

    # Generate timestamp for non-cached LLM call
    current_timestamp = int(time.time())
    return remove_think_tags(res), current_timestamp


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """清洗并标准化提取的文本
    参数：
        input_text：待处理的文本字符串
        is_name：输入文本是否为实体名称或关系名称

    返回：
        经过清洗和标准化的文本字符串
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """
    规范化实体/关系名称与描述，主要包括以下规则：
    - 清理HTML标签（段落和换行）
    - 中文全角符号转为半角符号
    - 删除中文之间的空格
    - 删除中英文/数字之间的空格
    - 保留英文字母和数字内部的空格
    - 中文括号替换为英文括号
    - 中文破折号替换为英文破折号
    - 移除文本首尾的英文引号
    - 移除英文引号环绕/嵌入在中文两侧
    - 移除中文引号
    - 过滤长度小于3且全为数字/小数点的内容
    - remove_inner_quotes=True 时：
        - 移除中文引号
        - 移除嵌入在中文中的英文引号
        - 不间断空格转普通空格
        - 数字后较窄的不间断空格转普通空格

    参数:
        name: 待规范化文本
        remove_inner_quotes: 是否移除内部引号及特殊空格

    返回:
        规范化后的文本字符串
    """
    # 移除HTML标签 <p> 和 <br>
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # 全角大/小写字母转半角
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # 全角数字转半角
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # 全角符号转半角符号
    name = name.replace("－", "-")      # 全角减号
    name = name.replace("＋", "+")      # 全角加号
    name = name.replace("／", "/")      # 全角斜线
    name = name.replace("＊", "*")      # 全角星号

    # 中文括号转英文括号
    name = name.replace("（", "(").replace("）", ")")

    # 中文破折号/全角破折号转英文破折号
    name = name.replace("—", "-").replace("－", "-")

    # 全角空格转普通空格
    name = name.replace("　", " ")

    # 删除中文之间的空格
    # (?<=[\u4e00-\u9fa5]) 匹配前一个是中文；(?=[\u4e00-\u9fa5]) 匹配后一个是中文
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # 删除中文和英文/数字/特殊符号之间的空格
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # 移除包裹文本的中英文引号
    if len(name) >= 2:
        # 英文双引号
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:
                name = inner_content

        # 英文单引号
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:
                name = inner_content

        # 中文双引号
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        # 中文单引号
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        # 书名号
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # 移除所有中文引号
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # 移除嵌在中文内外的英文引号
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        # 不间断空格(\u00a0)转普通空格
        name = name.replace("\u00a0", " ")
        # 数字后出现的窄不间断空格(\u202F)转普通空格（如“条款 3.1 内容”）
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    # 移除首尾空白符
    name = name.strip()

    # 过滤长度小于3且内容为纯数字的字符串
    # if len(name) < 3 and re.match(r"^[0-9]+$", name):
    #     return ""

    def should_filter_by_dots(text):
        """
        检查字符串是否仅由数字和点号组成，且至少包含一个点号。
        用于如: 1.2.3, 12.3, .123, 123., 12.3., .1.23 等情况。
        """
        return all(c.isdigit() or c == "." for c in text) and "." in text

    # if len(name) < 6 and should_filter_by_dots(name):
    #     # 长度小于6且为混合数字和点号的内容，过滤掉
    #     return ""

    return name


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """
    对输入文本进行清理，确保能安全地进行 UTF-8 编码。具体步骤如下：
      - 移除或替换代理字符（代理区：U+D800 至 U+DFFF），这些是编码错误的主要根源；
      - 移除或替换其他非法 Unicode 序列（如 U+FFFE / U+FFFF）；
      - 反转义 HTML 实体符号为正常字符；
      - 移除大多数控制字符（仅保留常见空白符：\t, \n, \r）；
      - 移除文本首尾的空白字符；
      - 若无法完全清理非法字符，则抛出 ValueError 异常而不是返回占位符。

    参数:
        text: 输入的待清理文本
        replacement_char: 用于替换不可接受字符的字符，默认使用空字符串

    返回:
        清理后的、可以安全进行 UTF-8 编码的文本字符串

    异常:
        ValueError: 输入文本存在无法修复的编码问题时抛出此异常，便于调用方处理

    说明:
        - 避免返回肉眼难以发现的非法字符相关的文本，保证最终下游处理安全。
        - 推荐 replacement_char 使用空字符串，若想保留非法字符位置可考虑使用占位符。
    """
    if not text:
        return text

    try:
        # 移除文本两端空白字符
        text = text.strip()

        if not text:
            return text

        # 尝试直接编码，捕捉潜在的编码问题
        text.encode("utf-8")

        # 处理代理区字符（U+D800 至 U+DFFF）
        sanitized = ""
        for char in text:
            code_point = ord(char)
            if 0xD800 <= code_point <= 0xDFFF:
                # 代理字符，替换为 replacement_char
                sanitized += replacement_char
                continue
            elif code_point == 0xFFFE or code_point == 0xFFFF:
                # Unicode 非字符，替换
                sanitized += replacement_char
                continue
            else:
                sanitized += char

        # 移除 null 字节和大多数控制字符（保留常用空白符 \t \n \r）
        sanitized = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", replacement_char, sanitized
        )

        # 再次尝试编码，确保安全
        sanitized.encode("utf-8")

        # HTML 实体反转义，例如 &amp; --> &
        sanitized = html.unescape(sanitized)

        # 移除所有控制字符，包括 C1 区间（\x7F-\x9F），但保留常见空白
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", sanitized)

        # 最终结果移除首尾空白
        return sanitized.strip()

    except UnicodeEncodeError as e:
        # 捕获不可清理的编码问题
        error_msg = f"文本包含无法清理的 UTF-8 编码问题: {str(e)[:100]}"
        logger.error(f"文本清理失败: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        # 其它异常类型，先尝试原文能否直接编码
        logger.error(f"文本清理出现未知错误: {str(e)}")
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            raise ValueError(
                f"文本清理时出现异常，且无法安全编码: {str(e)}"
            ) from e


def check_storage_env_vars(storage_name: str) -> None:
    """Check if all required environment variables for storage implementation exist

    Args:
        storage_name: Storage implementation name

    Raises:
        ValueError: If required environment variables are missing
    """
    from .kg import STORAGE_ENV_REQUIREMENTS

    required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        raise ValueError(
            f"Storage implementation '{storage_name}' requires the following "
            f"environment variables: {', '.join(missing_vars)}"
        )


def pick_by_weighted_polling(
    entities_or_relations: list[dict],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """
    Linear gradient weighted polling algorithm for text chunk selection.

    This algorithm ensures that entities/relations with higher importance get more text chunks,
    forming a linear decreasing allocation pattern.

    Args:
        entities_or_relations: List of entities or relations sorted by importance (high to low)
        max_related_chunks: Expected number of text chunks for the highest importance entity/relation
        min_related_chunks: Expected number of text chunks for the lowest importance entity/relation

    Returns:
        List of selected text chunk IDs
    """
    if not entities_or_relations:
        return []

    n = len(entities_or_relations)
    if n == 1:
        # Only one entity/relation, return its first max_related_chunks text chunks
        entity_chunks = entities_or_relations[0].get("sorted_chunks", [])
        return entity_chunks[:max_related_chunks]

    # Calculate expected text chunk count for each position (linear decrease)
    expected_counts = []
    for i in range(n):
        # Linear interpolation: from max_related_chunks to min_related_chunks
        ratio = i / (n - 1) if n > 1 else 0
        expected = max_related_chunks - ratio * (
            max_related_chunks - min_related_chunks
        )
        expected_counts.append(int(round(expected)))

    # First round allocation: allocate by expected values
    selected_chunks = []
    used_counts = []  # Track number of chunks used by each entity
    total_remaining = 0  # Accumulate remaining quotas

    for i, entity_rel in enumerate(entities_or_relations):
        entity_chunks = entity_rel.get("sorted_chunks", [])
        expected = expected_counts[i]

        # Actual allocatable count
        actual = min(expected, len(entity_chunks))
        selected_chunks.extend(entity_chunks[:actual])
        used_counts.append(actual)

        # Accumulate remaining quota
        remaining = expected - actual
        if remaining > 0:
            total_remaining += remaining

    # Second round allocation: multi-round scanning to allocate remaining quotas
    for _ in range(total_remaining):
        allocated = False

        # Scan entities one by one, allocate one chunk when finding unused chunks
        for i, entity_rel in enumerate(entities_or_relations):
            entity_chunks = entity_rel.get("sorted_chunks", [])

            # Check if there are still unused chunks
            if used_counts[i] < len(entity_chunks):
                # Allocate one chunk
                selected_chunks.append(entity_chunks[used_counts[i]])
                used_counts[i] += 1
                allocated = True
                break

        # If no chunks were allocated in this round, all entities are exhausted
        if not allocated:
            break

    return selected_chunks


async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    entity_info: list[dict[str, Any]],
    embedding_func: callable,
    query_embedding=None,
) -> list[str]:
    """
    Vector similarity-based text chunk selection algorithm.

    This algorithm selects text chunks based on cosine similarity between
    the query embedding and text chunk embeddings.

    Args:
        query: User's original query string
        text_chunks_storage: Text chunks storage instance
        chunks_vdb: Vector database storage for chunks
        num_of_chunks: Number of chunks to select
        entity_info: List of entity information containing chunk IDs
        embedding_func: Embedding function to compute query embedding

    Returns:
        List of selected text chunk IDs sorted by similarity (highest first)
    """
    logger.debug(
        f"Vector similarity chunk selection: num_of_chunks={num_of_chunks}, entity_info_count={len(entity_info) if entity_info else 0}"
    )

    if not entity_info or num_of_chunks <= 0:
        return []

    # Collect all unique chunk IDs from entity info
    all_chunk_ids = set()
    for i, entity in enumerate(entity_info):
        chunk_ids = entity.get("sorted_chunks", [])
        all_chunk_ids.update(chunk_ids)

    if not all_chunk_ids:
        logger.warning(
            "Vector similarity chunk selection:  no chunk IDs found in entity_info"
        )
        return []

    logger.debug(
        f"Vector similarity chunk selection: {len(all_chunk_ids)} unique chunk IDs collected"
    )

    all_chunk_ids = list(all_chunk_ids)

    try:
        # Use pre-computed query embedding if provided, otherwise compute it
        if query_embedding is None:
            query_embedding = await embedding_func([query])
            query_embedding = query_embedding[
                0
            ]  # Extract first embedding from batch result
            logger.debug(
                "Computed query embedding for vector similarity chunk selection"
            )
        else:
            logger.debug(
                "Using pre-computed query embedding for vector similarity chunk selection"
            )

        # Get chunk embeddings from vector database
        chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
        logger.debug(
            f"Vector similarity chunk selection: {len(chunk_vectors)} chunk vectors Retrieved"
        )

        if not chunk_vectors or len(chunk_vectors) != len(all_chunk_ids):
            if not chunk_vectors:
                logger.warning(
                    "Vector similarity chunk selection: no vectors retrieved from chunks_vdb"
                )
            else:
                logger.warning(
                    f"Vector similarity chunk selection: found {len(chunk_vectors)} but expecting {len(all_chunk_ids)}"
                )
            return []

        # Calculate cosine similarities
        similarities = []
        valid_vectors = 0
        
        # 一次性获取所有块的块元数据以提高性能
        chunk_metadata = {}
        if all_chunk_ids:
            try:
                chunks_data = await text_chunks_storage.get_by_ids(all_chunk_ids)
                chunk_metadata = {chunk['id']: chunk for chunk in chunks_data if chunk}
            except Exception as e:
                logger.warning(f"获取块元数据失败：{e}")
        
        # 获取当前日期，格式为YYYYMMDD
        current_date = datetime.now().strftime('%Y%m%d')
        
        for chunk_id in all_chunk_ids:
            if chunk_id in chunk_vectors:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    
                    # Check if chunk has abolition_date and apply penalty if needed
                    if chunk_id in chunk_metadata and 'abolition_date' in chunk_metadata[chunk_id]:
                        abolition_date = chunk_metadata[chunk_id]['abolition_date']
                        if abolition_date and abolition_date < current_date:
                            # Apply 0.7 penalty for abolished products
                            similarity *= 0.7
                            logger.debug(f"Applied abolition penalty to chunk {chunk_id}: similarity adjusted from {similarity/0.7:.4f} to {similarity:.4f}")
                    
                    similarities.append((chunk_id, similarity))
                    valid_vectors += 1
                except Exception as e:
                    logger.warning(
                        f"Vector similarity chunk selection: failed to calculate similarity for chunk {chunk_id}: {e}"
                    )
            else:
                logger.warning(
                    f"Vector similarity chunk selection:  no vector found for chunk {chunk_id}"
                )

        # Sort by similarity (highest first) and select top num_of_chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk_id for chunk_id, _ in similarities[:num_of_chunks]]

        logger.debug(
            f"Vector similarity chunk selection: {len(selected_chunks)} chunks from {len(all_chunk_ids)} candidates"
        )

        return selected_chunks

    except Exception as e:
        logger.error(f"[VECTOR_SIMILARITY] Error in vector similarity sorting: {e}")
        import traceback

        logger.error(f"[VECTOR_SIMILARITY] Traceback: {traceback.format_exc()}")
        # Fallback to simple truncation
        logger.debug("[VECTOR_SIMILARITY] Falling back to simple truncation")
        return all_chunk_ids[:num_of_chunks]


class TokenTracker:
    """Track token usage for LLM calls."""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts):
        """Add token usage from one LLM call.

        Args:
            token_counts: A dictionary containing prompt_tokens, completion_tokens, total_tokens
        """
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)

        # If total_tokens is provided, use it directly; otherwise calculate the sum
        if "total_tokens" in token_counts:
            self.total_tokens += token_counts["total_tokens"]
        else:
            self.total_tokens += token_counts.get(
                "prompt_tokens", 0
            ) + token_counts.get("completion_tokens", 0)

        self.call_count += 1

    def get_usage(self):
        """Get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def __str__(self):
        usage = self.get_usage()
        return (
            f"LLM call count: {usage['call_count']}, "
            f"Prompt tokens: {usage['prompt_tokens']}, "
            f"Completion tokens: {usage['completion_tokens']}, "
            f"Total tokens: {usage['total_tokens']}"
        )


async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_n: int = None,
) -> list[dict]:
    """
    Apply reranking to retrieved documents if rerank is enabled.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        global_config: Global configuration containing rerank settings
        enable_rerank: Whether to enable reranking from query parameter
        top_n: Number of top documents to return after reranking

    Returns:
        Reranked documents if rerank is enabled, otherwise original documents
    """
    if not enable_rerank or not retrieved_docs:
        return retrieved_docs

    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters."
        )
        return retrieved_docs

    try:
        # Extract document content for reranking
        document_texts = []
        for doc in retrieved_docs:
            # Try multiple possible content fields
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("chunk_content")
                or doc.get("document")
                or str(doc)
            )
            document_texts.append(content)

        # Call the new rerank function that returns index-based results
        rerank_results = await rerank_func(
            query=query,
            documents=document_texts,
            top_n=top_n,
        )

        # Process rerank results based on return format
        if rerank_results and len(rerank_results) > 0:
            # Check if results are in the new index-based format
            if isinstance(rerank_results[0], dict) and "index" in rerank_results[0]:
                # New format: [{"index": 0, "relevance_score": 0.85}, ...]
                reranked_docs = []
                for result in rerank_results:
                    index = result["index"]
                    relevance_score = result["relevance_score"]

                    # Get original document and add rerank score
                    if 0 <= index < len(retrieved_docs):
                        doc = retrieved_docs[index].copy()
                        doc["rerank_score"] = relevance_score
                        reranked_docs.append(doc)

                logger.info(
                    f"Successfully reranked: {len(reranked_docs)} chunks from {len(retrieved_docs)} original chunks"
                )
                return reranked_docs
            else:
                # Legacy format: assume it's already reranked documents
                logger.info(f"Using legacy rerank format: {len(rerank_results)} chunks")
                return rerank_results[:top_n] if top_n else rerank_results
        else:
            logger.warning("Rerank returned empty results, using original chunks")
            return retrieved_docs

    except Exception as e:
        logger.error(f"Error during reranking: {e}, using original chunks")
        return retrieved_docs


async def process_chunks_unified(
    query: str,
    unique_chunks: list[dict],
    query_param: "QueryParam",
    global_config: dict,
    source_type: str = "mixed",
    chunk_token_limit: int = None,  # Add parameter for dynamic token limit
) -> list[dict]:
    """
    Unified processing for text chunks: deduplication, chunk_top_k limiting, reranking, and token truncation.

    Args:
        query: Search query for reranking
        chunks: List of text chunks to process
        query_param: Query parameters containing configuration
        global_config: Global configuration dictionary
        source_type: Source type for logging ("vector", "entity", "relationship", "mixed")
        chunk_token_limit: Dynamic token limit for chunks (if None, uses default)

    Returns:
        Processed and filtered list of text chunks
    """
    if not unique_chunks:
        return []

    origin_count = len(unique_chunks)

    # 1. Apply reranking if enabled and query is provided
    if query_param.enable_rerank and query and unique_chunks:
        rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_n=rerank_top_k,
        )

    # 2. Filter by minimum rerank score if reranking is enabled
    if query_param.enable_rerank and unique_chunks:
        min_rerank_score = global_config.get("min_rerank_score", 0.5)
        if min_rerank_score > 0.0:
            original_count = len(unique_chunks)

            # Filter chunks with score below threshold
            filtered_chunks = []
            for chunk in unique_chunks:
                rerank_score = chunk.get(
                    "rerank_score", 1.0
                )  # Default to 1.0 if no score
                if rerank_score >= min_rerank_score:
                    filtered_chunks.append(chunk)

            unique_chunks = filtered_chunks
            filtered_count = original_count - len(unique_chunks)

            if filtered_count > 0:
                logger.info(
                    f"Rerank filtering: {len(unique_chunks)} chunks remained (min rerank score: {min_rerank_score})"
                )
            if not unique_chunks:
                return []

    # 3. Apply chunk_top_k limiting if specified
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
        logger.debug(
            f"Kept chunk_top-k: {len(unique_chunks)} chunks (deduplicated original: {origin_count})"
        )

    # 4. Token-based final truncation
    tokenizer = global_config.get("tokenizer")
    if tokenizer and unique_chunks:
        # Set default chunk_token_limit if not provided
        if chunk_token_limit is None:
            # Get default from query_param or global_config
            chunk_token_limit = getattr(
                query_param,
                "max_total_tokens",
                global_config.get("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS),
            )

        original_count = len(unique_chunks)

        unique_chunks = truncate_list_by_token_size(
            unique_chunks,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=chunk_token_limit,
            tokenizer=tokenizer,
        )

        logger.debug(
            f"Token truncation: {len(unique_chunks)} chunks from {original_count} "
            f"(chunk available tokens: {chunk_token_limit}, source: {source_type})"
        )

    # 5. add id field to each chunk
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        chunk_with_id["id"] = f"DC{i + 1}"
        final_chunks.append(chunk_with_id)

    return final_chunks


def build_file_path(already_file_paths, data_list, target):
    """Build file path string with UTF-8 byte length limit and deduplication

    Args:
        already_file_paths: List of existing file paths
        data_list: List of data items containing file_path
        target: Target name for logging warnings

    Returns:
        str: Combined file paths separated by GRAPH_FIELD_SEP
    """
    # set: deduplication
    file_paths_set = {fp for fp in already_file_paths if fp}

    # string: filter empty value and keep file order in already_file_paths
    file_paths = GRAPH_FIELD_SEP.join(fp for fp in already_file_paths if fp)

    # Check if initial file_paths already exceeds byte length limit
    if len(file_paths.encode("utf-8")) >= DEFAULT_MAX_FILE_PATH_LENGTH:
        logger.warning(
            f"Initial file_paths already exceeds {DEFAULT_MAX_FILE_PATH_LENGTH} bytes for {target}, "
            f"current size: {len(file_paths.encode('utf-8'))} bytes"
        )

    # ignored file_paths
    file_paths_ignore = ""
    # add file_paths
    for dp in data_list:
        cur_file_path = dp.get("file_path")
        # empty
        if not cur_file_path:
            continue

        # skip duplicate item
        if cur_file_path in file_paths_set:
            continue
        # add
        file_paths_set.add(cur_file_path)

        # check the UTF-8 byte length
        new_addition = GRAPH_FIELD_SEP + cur_file_path if file_paths else cur_file_path
        if (
            len(file_paths.encode("utf-8")) + len(new_addition.encode("utf-8"))
            < DEFAULT_MAX_FILE_PATH_LENGTH - 5
        ):
            # append
            file_paths += new_addition
        else:
            # ignore
            file_paths_ignore += GRAPH_FIELD_SEP + cur_file_path

    if file_paths_ignore:
        logger.warning(
            f"File paths exceed {DEFAULT_MAX_FILE_PATH_LENGTH} bytes for {target}, "
            f"ignoring file path: {file_paths_ignore}"
        )
    return file_paths


def generate_track_id(prefix: str = "upload") -> str:
    """生成一个包含时间戳和UUID的唯一跟踪ID

    参数：
        prefix：跟踪ID的前缀（例如：'upload'、'insert'）

    返回：
        str：格式为{prefix}_{timestamp}_{uuid}的唯一跟踪ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # 使用UUID的前8个字符
    return f"{prefix}_{timestamp}_{unique_id}"


def get_pinyin_sort_key(text: str) -> str:
    """生成用于中文拼音排序的排序键

    此函数使用pypinyin实现真正的中文拼音排序。
    如果pypinyin不可用，则退回到简单的小写字符串排序。

    参数：
        text：要为其生成排序键的文本

    返回：
        str：可用于比较和排序的排序键
    """
    if not text:
        return ""

    if _PYPINYIN_AVAILABLE:
        try:
            # Convert Chinese characters to pinyin, keep non-Chinese as-is
            pinyin_list = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
            return "".join(pinyin_list).lower()
        except Exception:
            # Silently fall back to simple string sorting on any error
            return text.lower()
    else:
        # pypinyin not available, use simple string sorting
        return text.lower()


def fix_tuple_delimiter_corruption(
    record: str, delimiter_core: str, tuple_delimiter: str
) -> str:
    """
    Fix various forms of tuple_delimiter corruption from LLM output.

    This function handles missing or replaced characters around the core delimiter.
    It fixes common corruption patterns where the LLM output doesn't match the expected
    tuple_delimiter format.

    Args:
        record: The text record to fix
        delimiter_core: The core delimiter (e.g., "S" from "<|#|>")
        tuple_delimiter: The complete tuple delimiter (e.g., "<|#|>")

    Returns:
        The corrected record with proper tuple_delimiter format
    """
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    # Escape the delimiter core for regex use
    escaped_delimiter_core = re.escape(delimiter_core)

    # Fix: <|##|> -> <|#|>, <|#||#|> -> <|#|>, <|#|||#|> -> <|#|>
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|\#|> -> <|#|>
    record = re.sub(
        rf"<\|\\{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|> -> <|#|>, <||> -> <|#|>
    record = re.sub(
        r"<\|+>",
        tuple_delimiter,
        record,
    )

    # Fix: <X|#|> -> <|#|>, <|#|Y> -> <|#|>, <X|#|Y> -> <|#|>, <||#||> -> <|#|>, <||#> -> <|#|> (one extra characters outside pipes)
    record = re.sub(
        rf"<.?\|{escaped_delimiter_core}\|*?>",
        tuple_delimiter,
        record,
    )

    # Fix: <#>, <#|>, <|#> -> <|#|> (missing one or both pipes)
    record = re.sub(
        rf"<\|?{escaped_delimiter_core}\|?>",
        tuple_delimiter,
        record,
    )

    # Fix: <X#|> -> <|#|>, <|#X> -> <|#|> (one pipe is replaced by other character)
    record = re.sub(
        rf"<[^|]{escaped_delimiter_core}\|>|<\|{escaped_delimiter_core}[^|]>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#| -> <|#|>, <|#|| -> <|#|> (missing closing >)

    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|+(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix <|#: -> <|#|> (missing closing >)
    record = re.sub(
        rf"<\|{escaped_delimiter_core}:(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: <|| -> <|#|>
    record = re.sub(
        r"<\|\|(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: |#|> -> <|#|> (missing opening <)
    record = re.sub(
        rf"(?<!<)\|{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#|>| -> <|#|>  ( this is a fix for: <|#|| -> <|#|> )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|>\|",
        tuple_delimiter,
        record,
    )

    # Fix: ||#|| -> <|#|> (double pipes on both sides without angle brackets)
    record = re.sub(
        rf"\|\|{escaped_delimiter_core}\|\|",
        tuple_delimiter,
        record,
    )

    return record


def create_prefixed_exception(original_exception: Exception, prefix: str) -> Exception:
    """
    Safely create a prefixed exception that adapts to all error types.

    Args:
        original_exception: The original exception.
        prefix: The prefix to add.

    Returns:
        A new exception with the prefix, maintaining the original exception type if possible.
    """
    try:
        # Method 1: Try to reconstruct using original arguments.
        if hasattr(original_exception, "args") and original_exception.args:
            args = list(original_exception.args)
            # Find the first string argument and prefix it. This is safer for
            # exceptions like OSError where the first arg is an integer (errno).
            found_str = False
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    args[i] = f"{prefix}: {arg}"
                    found_str = True
                    break

            # If no string argument is found, prefix the first argument's string representation.
            if not found_str:
                args[0] = f"{prefix}: {args[0]}"

            return type(original_exception)(*args)
        else:
            # Method 2: If no args, try single parameter construction.
            return type(original_exception)(f"{prefix}: {str(original_exception)}")
    except (TypeError, ValueError, AttributeError) as construct_error:
        # Method 3: If reconstruction fails, wrap it in a RuntimeError.
        # This is the safest fallback, as attempting to create the same type
        # with a single string can fail if the constructor requires multiple arguments.
        return RuntimeError(
            f"{prefix}: {type(original_exception).__name__}: {str(original_exception)} "
            f"(Original exception could not be reconstructed: {construct_error})"
        )


def convert_to_user_format(
    entities_context: list[dict],
    relations_context: list[dict],
    chunks: list[dict],
    references: list[dict],
    query_mode: str,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
) -> dict[str, Any]:
    """Convert internal data format to user-friendly format using original database data"""

    # Convert entities format using original data when available
    formatted_entities = []
    for entity in entities_context:
        entity_name = entity.get("entity", "")

        # Try to get original data first
        original_entity = None
        if entity_id_to_original and entity_name in entity_id_to_original:
            original_entity = entity_id_to_original[entity_name]

        if original_entity:
            # Use original database data
            formatted_entities.append(
                {
                    "entity_name": original_entity.get("entity_name", entity_name),
                    "entity_type": original_entity.get("entity_type", "UNKNOWN"),
                    "description": original_entity.get("description", ""),
                    "source_id": original_entity.get("source_id", ""),
                    "file_path": original_entity.get("file_path", "unknown_source"),
                    "created_at": original_entity.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", "unknown_source"),
                    "created_at": entity.get("created_at", ""),
                }
            )

    # Convert relationships format using original data when available
    formatted_relationships = []
    for relation in relations_context:
        entity1 = relation.get("entity1", "")
        entity2 = relation.get("entity2", "")
        relation_key = (entity1, entity2)

        # Try to get original data first
        original_relation = None
        if relation_id_to_original and relation_key in relation_id_to_original:
            original_relation = relation_id_to_original[relation_key]

        if original_relation:
            # Use original database data
            formatted_relationships.append(
                {
                    "src_id": original_relation.get("src_id", entity1),
                    "tgt_id": original_relation.get("tgt_id", entity2),
                    "description": original_relation.get("description", ""),
                    "keywords": original_relation.get("keywords", ""),
                    "weight": original_relation.get("weight", 1.0),
                    "source_id": original_relation.get("source_id", ""),
                    "file_path": original_relation.get("file_path", "unknown_source"),
                    "created_at": original_relation.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_relationships.append(
                {
                    "src_id": entity1,
                    "tgt_id": entity2,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": relation.get("weight", 1.0),
                    "source_id": relation.get("source_id", ""),
                    "file_path": relation.get("file_path", "unknown_source"),
                    "created_at": relation.get("created_at", ""),
                }
            )

    # Convert chunks format (chunks already contain complete data)
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
            "file_path": chunk.get("file_path", "unknown_source"),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        formatted_chunks.append(chunk_data)

    logger.debug(
        f"[convert_to_user_format] Formatted {len(formatted_chunks)}/{len(chunks)} chunks"
    )

    # Build basic metadata (metadata details will be added by calling functions)
    metadata = {
        "query_mode": query_mode,
        "keywords": {
            "high_level": [],
            "low_level": [],
        },  # Placeholder, will be set by calling functions
    }

    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "chunks": formatted_chunks,
            "references": references,
        },
        "metadata": metadata,
    }


def generate_reference_list_from_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Generate reference list from chunks, prioritizing by occurrence frequency.

    This function extracts file_paths from chunks, counts their occurrences,
    sorts by frequency and first appearance order, creates reference_id mappings,
    and builds a reference_list structure.

    Args:
        chunks: List of chunk dictionaries with file_path information

    Returns:
        tuple: (reference_list, updated_chunks_with_reference_ids)
            - reference_list: List of dicts with reference_id and file_path
            - updated_chunks_with_reference_ids: Original chunks with reference_id field added
    """
    if not chunks:
        return [], []

    # 1. Extract all valid file_paths and count their occurrences
    file_path_counts = {}
    for chunk in chunks:
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source":
            file_path_counts[file_path] = file_path_counts.get(file_path, 0) + 1

    # 2. Sort file paths by frequency (descending), then by first appearance order
    # Create a list of (file_path, count, first_index) tuples
    file_path_with_indices = []
    seen_paths = set()
    for i, chunk in enumerate(chunks):
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source" and file_path not in seen_paths:
            file_path_with_indices.append((file_path, file_path_counts[file_path], i))
            seen_paths.add(file_path)

    # Sort by count (descending), then by first appearance index (ascending)
    sorted_file_paths = sorted(file_path_with_indices, key=lambda x: (-x[1], x[2]))
    unique_file_paths = [item[0] for item in sorted_file_paths]

    # 3. Create mapping from file_path to reference_id (prioritized by frequency)
    file_path_to_ref_id = {}
    for i, file_path in enumerate(unique_file_paths):
        file_path_to_ref_id[file_path] = str(i + 1)

    # 4. Add reference_id field to each chunk
    updated_chunks = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        file_path = chunk_copy.get("file_path", "")
        if file_path and file_path != "unknown_source":
            chunk_copy["reference_id"] = file_path_to_ref_id[file_path]
        else:
            chunk_copy["reference_id"] = ""
        updated_chunks.append(chunk_copy)

    # 5. Build reference_list
    reference_list = []
    for i, file_path in enumerate(unique_file_paths):
        reference_list.append({"reference_id": str(i + 1), "file_path": file_path})

    return reference_list, updated_chunks
