"""
Light_Graph_RAG 配置常量集中定义

本模块集中定义了 Light_Graph_RAG 系统在不同部分使用的默认配置常量。
通过集中配置，确保各部分一致性和便于维护。

重要说明：
-----------
这些常量提供默认值，可以通过 .env 文件中的环境变量进行覆盖。

参数优先级：
1. ✅ 最高优先级：.env 文件中的环境变量
2. 📦 默认优先级：本文件中定义的 DEFAULT_* 常量

示例：
如果在 .env 中设置 TOP_K=60，则会覆盖本文件中的 DEFAULT_TOP_K=40。

详细配置说明请参考：ENV_CONFIG_GUIDE.md
"""

# 服务器参数默认值
DEFAULT_WOKERS = 2  # Gunicorn 默认工作进程数
DEFAULT_MAX_GRAPH_NODES = 1000  # 知识图谱允许的最大节点数

# 信息抽取相关默认值
DEFAULT_SUMMARY_LANGUAGE = "Chinese"  # 文档处理默认使用的语言
DEFAULT_MAX_GLEANING = 1  # 默认最多梳理轮次

# 触发LLM总结的描述片段数量与最大令牌数
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8  # 描述片段数量达到该值则触发 LLM 总结
DEFAULT_SUMMARY_MAX_TOKENS = 1200       # 触发 LLM 总结时允许的最大描述令牌数量
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600  # 推荐的 LLM 输出摘要字数（单位：token）
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000    # 总结时传给 LLM 的最大上下文 token 数

# 如果 .env 未指定 ENTITY_TYPES 时的默认实体类型
DEFAULT_ENTITY_TYPES = ["保险产品", "保险条款", "保险概念", "保险责任", "数据", "期限", "角色", "机构", "疾病", "事件", "数据型表格", "概念型表格"]

# 知识图谱字段分隔符
GRAPH_FIELD_SEP = "<SEP>"

# 查询和检索参数默认值
DEFAULT_TOP_K = 40             # 全局召回召回数
DEFAULT_CHUNK_TOP_K = 20       # 文档片段召回 top-K
DEFAULT_MAX_ENTITY_TOKENS = 6000      # 实体最大 token 数
DEFAULT_MAX_RELATION_TOKENS = 8000    # 关系描述最大 token 数
DEFAULT_MAX_TOTAL_TOKENS = 30000      # LLM 单轮最大可处理 token 数
DEFAULT_COSINE_THRESHOLD = 0.2        # 向量检索相似度阈值
DEFAULT_RELATED_CHUNK_NUMBER = 5      # 相关文档片段默认数量
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"  # 知识片段选择方法

# 历史轮次，仅作兼容保留，所有历史消息已传递给 LLM
DEFAULT_HISTORY_TURNS = 0  # TODO: 已废弃

# 重排序参数默认值
DEFAULT_MIN_RERANK_SCORE = 0.0  # 最小 rerank 分数
DEFAULT_RERANK_BINDING = "null" # rerank 绑定参数（保留）

# 向量和图数据库的文件路径参数（不可修改，Milvus Schema 用到）
DEFAULT_MAX_FILE_PATH_LENGTH = 32768  # 文件路径最大长度

# LLM 默认温度参数
DEFAULT_TEMPERATURE = 1.0

# 异步处理相关默认值
DEFAULT_MAX_ASYNC = 4  # 最大异步执行数量
DEFAULT_MAX_PARALLEL_INSERT = 2  # 最大并发插入操作数

# 向量嵌入相关配置默认值
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # 嵌入计算最大异步数量
DEFAULT_EMBEDDING_BATCH_NUM = 10      # 嵌入批处理默认数量

# Gunicorn 服务超时时间
DEFAULT_TIMEOUT = 300  # 单位：秒

# LLM 及嵌入超时时间（秒）
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# 日志配置默认值
DEFAULT_LOG_MAX_BYTES = 10485760  # 日志文件最大 10MB
DEFAULT_LOG_BACKUP_COUNT = 5      # 保留日志备份数量
DEFAULT_LOG_FILENAME = "lightrag.log"  # 默认日志文件名

# Ollama 服务默认参数
DEFAULT_OLLAMA_MODEL_NAME = "lightrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = "2024-01-15T00:00:00Z"
DEFAULT_OLLAMA_DIGEST = "sha256:lightrag"
