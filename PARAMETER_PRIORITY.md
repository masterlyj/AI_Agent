# LightRAG 参数优先级说明

## 📊 优先级层次

```
┌─────────────────────────────────────────┐
│   优先级 1: .env 环境变量（最高）      │  ⭐ 推荐配置位置
│   在 .env 文件中设置的所有参数        │
└─────────────────────────────────────────┘
                   ↓
                覆盖
                   ↓
┌─────────────────────────────────────────┐
│   优先级 2: constants.py 默认常量       │  🔧 开箱即用的默认值
│   提供合理的默认配置                    │
└─────────────────────────────────────────┘
```

## ⚡ 核心原则

1. **`.env` 文件优先级最高**
   - 在 `.env` 中设置的任何参数都会覆盖 `constants.py` 中的默认值
   - 这是调整配置的**推荐方式**，无需修改代码

2. **`constants.py` 作为兜底默认值**
   - 如果 `.env` 中没有设置某个参数，系统自动使用 `constants.py` 中的默认值
   - 这些默认值已经过调优，适合大多数使用场景

## 🎯 实际示例

### 示例 1：查询召回数配置

**constants.py 中的默认值：**
```python
DEFAULT_TOP_K = 40  # 全局召回数
```

**三种使用方式：**

#### 方式 A：完全使用默认值（不设置 .env）
```bash
# .env 文件中不设置 TOP_K
```
**结果**：系统使用 `TOP_K = 40`

---

#### 方式 B：通过 .env 覆盖（推荐）
```bash
# .env 文件中设置
TOP_K=60
```
**结果**：系统使用 `TOP_K = 60` ✅（覆盖了默认值）

---

#### 方式 C：动态调整
当你需要临时测试不同配置时，只需修改 `.env` 文件，无需改动代码：
```bash
# 测试配置 1
TOP_K=80

# 测试配置 2
TOP_K=100
```

## 📋 完整参数映射表

| .env 环境变量 | constants.py 默认常量 | 默认值 | 说明 |
|---------------|----------------------|--------|------|
| `WORKERS` | `DEFAULT_WOKERS` | 2 | Gunicorn 工作进程数 |
| `MAX_GRAPH_NODES` | `DEFAULT_MAX_GRAPH_NODES` | 1000 | 知识图谱最大节点数 |
| `SUMMARY_LANGUAGE` | `DEFAULT_SUMMARY_LANGUAGE` | "Chinese" | 文档处理语言 |
| `MAX_GLEANING` | `DEFAULT_MAX_GLEANING` | 1 | 实体抽取迭代次数 |
| `FORCE_LLM_SUMMARY_ON_MERGE` | `DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE` | 8 | 触发 LLM 总结的片段数 |
| `SUMMARY_MAX_TOKENS` | `DEFAULT_SUMMARY_MAX_TOKENS` | 1200 | 总结最大令牌数 |
| `SUMMARY_LENGTH_RECOMMENDED` | `DEFAULT_SUMMARY_LENGTH_RECOMMENDED` | 600 | 推荐摘要长度 |
| `SUMMARY_CONTEXT_SIZE` | `DEFAULT_SUMMARY_CONTEXT_SIZE` | 12000 | 总结上下文大小 |
| `ENTITY_TYPES` | `DEFAULT_ENTITY_TYPES` | \[11个类型\] | 实体类型列表 |
| `TOP_K` | `DEFAULT_TOP_K` | 40 | 全局召回数 |
| `CHUNK_TOP_K` | `DEFAULT_CHUNK_TOP_K` | 20 | 文档片段召回数 |
| `MAX_ENTITY_TOKENS` | `DEFAULT_MAX_ENTITY_TOKENS` | 6000 | 实体最大 token 数 |
| `MAX_RELATION_TOKENS` | `DEFAULT_MAX_RELATION_TOKENS` | 8000 | 关系最大 token 数 |
| `MAX_TOTAL_TOKENS` | `DEFAULT_MAX_TOTAL_TOKENS` | 30000 | LLM 单轮最大 token |
| `COSINE_THRESHOLD` | `DEFAULT_COSINE_THRESHOLD` | 0.2 | 向量相似度阈值 |
| `RELATED_CHUNK_NUMBER` | `DEFAULT_RELATED_CHUNK_NUMBER` | 5 | 相关片段数量 |
| `KG_CHUNK_PICK_METHOD` | `DEFAULT_KG_CHUNK_PICK_METHOD` | "VECTOR" | 片段选择方法 |
| `MIN_RERANK_SCORE` | `DEFAULT_MIN_RERANK_SCORE` | 0.0 | 最小 rerank 分数 |
| `MAX_ASYNC` | `DEFAULT_MAX_ASYNC` | 4 | 最大异步执行数 |
| `MAX_PARALLEL_INSERT` | `DEFAULT_MAX_PARALLEL_INSERT` | 2 | 最大并发插入数 |
| `EMBEDDING_FUNC_MAX_ASYNC` | `DEFAULT_EMBEDDING_FUNC_MAX_ASYNC` | 8 | 嵌入计算异步数 |
| `EMBEDDING_BATCH_NUM` | `DEFAULT_EMBEDDING_BATCH_NUM` | 10 | 嵌入批处理数量 |
| `TIMEOUT` | `DEFAULT_TIMEOUT` | 300 | 服务超时时间（秒）|
| `LLM_TIMEOUT` | `DEFAULT_LLM_TIMEOUT` | 180 | LLM 超时时间（秒）|
| `EMBEDDING_TIMEOUT` | `DEFAULT_EMBEDDING_TIMEOUT` | 30 | Embedding 超时（秒）|
| `LOG_MAX_BYTES` | `DEFAULT_LOG_MAX_BYTES` | 10485760 | 日志文件最大字节 |
| `LOG_BACKUP_COUNT` | `DEFAULT_LOG_BACKUP_COUNT` | 5 | 日志备份数量 |
| `LOG_FILENAME` | `DEFAULT_LOG_FILENAME` | "lightrag.log" | 日志文件名 |

## 🔧 代码实现原理

系统通过 `get_env_value()` 函数实现优先级机制：

```python
from .utils import get_env_value
from .constants import DEFAULT_TOP_K

# 读取配置：优先使用环境变量，否则使用默认值
top_k = get_env_value("TOP_K", DEFAULT_TOP_K, int)
```

**函数行为：**
1. 首先尝试从环境变量（`.env` 文件）读取 `TOP_K`
2. 如果环境变量存在，使用该值并转换为指定类型（`int`）
3. 如果环境变量不存在，使用 `DEFAULT_TOP_K` 作为默认值

## ✅ 推荐工作流程

### 第一次使用

1. 复制模板文件：
   ```bash
   cp env_template.txt .env
   ```

2. 保持大部分默认配置，只修改必需项：
   ```bash
   # .env 文件
   LLM_PROVIDER=google_genai
   LLM_MODEL=gemini-2.5-flash
   GOOGLE_API_KEY=your_api_key_here
   
   EMBEDDING_TYPE=ollama
   OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
   EMBEDDING_DIM=1024
   ```

3. 其他参数使用 `constants.py` 中的默认值

### 性能调优阶段

根据实际使用情况，在 `.env` 中逐步调整参数：

```bash
# 提高召回精度
TOP_K=60
CHUNK_TOP_K=30
MAX_ENTITY_TOKENS=8000

# 加快处理速度
MAX_ASYNC=8
MAX_PARALLEL_INSERT=4
```

### 生产环境部署

在 `.env` 中设置所有关键参数，确保配置明确：

```bash
# 明确设置所有性能相关参数
TOP_K=50
CHUNK_TOP_K=25
MAX_ENTITY_TOKENS=7000
MAX_RELATION_TOKENS=9000
MAX_TOTAL_TOKENS=35000

# 设置合理的超时时间
LLM_TIMEOUT=300
EMBEDDING_TIMEOUT=60
TIMEOUT=600

# 日志配置
LOG_MAX_BYTES=52428800  # 50MB
LOG_BACKUP_COUNT=10
```

## 🚫 注意事项

1. **不要直接修改 `constants.py`**
   - 保持 `constants.py` 不变，方便代码更新和维护
   - 所有自定义配置都应该放在 `.env` 文件中

2. **数据类型要匹配**
   - 数字参数使用数字：`TOP_K=60`（不要加引号）
   - 字符串参数使用文本：`SUMMARY_LANGUAGE=Chinese`
   - 布尔参数使用：`true` 或 `false`
   - 数组参数使用 JSON 格式：`ENTITY_TYPES=["Person", "Organization"]`

3. **特殊参数注意**
   - `ENTITY_TYPES` 必须是合法的 JSON 数组格式
   - `COSINE_THRESHOLD` 范围应在 0-1 之间
   - `WORKERS` 应根据 CPU 核心数设置（建议 2-4）

## 📚 参考文档

- **配置模板**：[env_template.txt](env_template.txt)
- **详细配置指南**：[ENV_CONFIG_GUIDE.md](ENV_CONFIG_GUIDE.md)
- **常量定义**：[src/Knowledge_Graph_Agent/constants.py](src/Knowledge_Graph_Agent/constants.py)

---

**总结**：`.env` 环境变量的优先级最高，它会覆盖 `constants.py` 中的所有默认值。这种设计让你可以在不修改代码的情况下灵活调整配置。

