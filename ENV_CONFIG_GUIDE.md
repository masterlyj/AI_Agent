# 环境变量配置指南

## 📋 概述

本项目已全面支持通过 `.env` 文件配置 LLM（大语言模型）、Embedding（嵌入模型）和 Rerank（精排模型）。所有模型配置均可通过环境变量进行管理，无需修改代码。

## 🚀 快速开始

### 1. 创建 .env 文件

在项目根目录下创建 `.env` 文件：

```bash
cp env_template.txt .env
```

### 2. 配置你的模型

根据你的需求编辑 `.env` 文件，填入相应的 API Key 和模型配置。

## 📝 配置说明

### LLM 大语言模型配置

支持 **Google Gemini** 和 **DeepSeek** 两种 LLM。

#### 环境变量

```bash
# 指定 LLM 提供商（可选值: google_genai | deepseek）
LLM_PROVIDER=google_genai

# 模型名称
# Google: gemini-2.5-flash, gemini-1.5-pro, etc.
# DeepSeek: deepseek-chat, deepseek-reasoner, etc.
LLM_MODEL=gemini-2.5-flash

# 温度参数 (0-1, 0 表示确定性最强)
LLM_TEMPERATURE=0

# API Keys
GOOGLE_API_KEY=your_google_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# DeepSeek API Base URL (一般不需要修改)
LLM_BASE_URL=https://api.deepseek.com/v1
```

#### 使用说明

- 如果设置了 `LLM_PROVIDER`，系统会优先使用指定的提供商
- 如果未设置 `LLM_PROVIDER`，系统会按优先级自动选择（Google > DeepSeek）
- 只需要配置你实际使用的 API Key

### Embedding 嵌入模型配置

支持三种 Embedding 来源：
1. **HuggingFace** (本地模型或 HuggingFace Hub)
2. **Ollama** (本地运行的 Ollama 服务)
3. **vLLM / OpenAI 兼容接口**

#### 环境变量

```bash
# 指定 Embedding 类型（可选值: hf | ollama | vllm）
EMBEDDING_TYPE=ollama

# Embedding 向量维度 (需要与实际模型匹配)
EMBEDDING_DIM=1024
```

#### 选项 1: HuggingFace

```bash
EMBEDDING_TYPE=hf

# 模型名称或本地路径
# 远程: BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2
# 本地: /path/to/local/model
HF_EMBEDDING_MODEL_NAME=BAAI/bge-m3

# 设备 (留空自动检测, 或指定 cuda, cpu, cuda:0 等)
HF_EMBEDDING_DEVICE=

# 其他选项
HF_EMBEDDING_TRUST_REMOTE_CODE=false
HF_EMBEDDING_SHOW_PROGRESS=false
HF_EMBEDDING_MULTI_PROCESS=false
```

#### 选项 2: Ollama（推荐）

```bash
EMBEDDING_TYPE=ollama

OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
OLLAMA_BASE_URL=http://localhost:11434
```

#### 选项 3: vLLM / OpenAI 兼容接口

```bash
EMBEDDING_TYPE=vllm

VLLM_EMBEDDING_MODEL=text-embedding-3-large
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

### Rerank 精排模型配置

Rerank 模型用于对检索结果进行二次精排，提高相关性。

#### 环境变量

```bash
# 是否启用精排
RERANK_ENABLED=false

# Rerank 模型名称或本地路径
# HuggingFace: maidalun1020/bce-reranker-base_v1, BAAI/bge-reranker-v2-m3
# 本地路径: /path/to/local/reranker/model
RERANK_MODEL=maidalun1020/bce-reranker-base_v1

# 设备 (留空自动检测, 或指定 cpu, cuda, cuda:0 等)
RERANK_DEVICE=

# 精排后返回的 Top K 文档数
RERANK_TOP_K=20

# 是否使用 FP16 精度 (可加速推理)
RERANK_USE_FP16=false
```

#### 使用本地模型

如果你想使用本地下载的模型，只需要将 `RERANK_MODEL` 设置为本地路径：

```bash
RERANK_MODEL=/path/to/your/local/reranker
```

代码会自动识别本地路径并从本地加载模型。

## 📦 使用示例

### Knowledge Graph Agent

```python
from src.Knowledge_Graph_Agent.agent import RAGAgent

# 所有配置从 .env 读取
agent = await RAGAgent.create()

# 索引文档
await agent.index_documents(["data/inputs/document.md"])

# 查询
result = await agent.query(
    "你的问题?",
    mode="hybrid",
    enable_rerank=True  # 如果 .env 中 RERANK_ENABLED=true，会使用精排
)
```

### Paper Study Agent

```python
from src.Paper_Study_Agent.app import PaperChatBot

# 不传入配置参数，自动从 .env 读取
chatbot = PaperChatBot(
    arxiv_ids=["2301.00001", "2302.00002"]
)

# 或者传入自定义配置覆盖 .env
chatbot = PaperChatBot(
    arxiv_ids=["2301.00001"],
    embedding_config={
        "type": "hf",
        "model_name": "BAAI/bge-m3"
    },
    rerank_config={
        "enabled": True,
        "model": "BAAI/bge-reranker-v2-m3"
    }
)
```

## 🔧 常见配置场景

### 场景 1: 使用 Google Gemini + Ollama Embedding

```bash
# .env
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_api_key

EMBEDDING_TYPE=ollama
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_DIM=1024

RERANK_ENABLED=false
```

### 场景 2: 使用 DeepSeek + HuggingFace 本地模型 + Rerank

```bash
# .env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_api_key

EMBEDDING_TYPE=hf
HF_EMBEDDING_MODEL_NAME=/path/to/local/bge-m3
HF_EMBEDDING_DEVICE=cuda:0
EMBEDDING_DIM=1024

RERANK_ENABLED=true
RERANK_MODEL=/path/to/local/reranker
RERANK_DEVICE=cuda:0
RERANK_TOP_K=20
```

### 场景 3: 全云端配置（适合开发测试）

```bash
# .env
LLM_PROVIDER=google_genai
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_api_key

EMBEDDING_TYPE=vllm
VLLM_EMBEDDING_MODEL=text-embedding-3-large
VLLM_BASE_URL=https://api.your-vllm-service.com/v1
VLLM_API_KEY=your_vllm_api_key
EMBEDDING_DIM=1536

RERANK_ENABLED=false
```

## 📂 工作目录配置

您还可以在 `.env` 中配置 RAG 系统的工作目录：

```bash
# RAG 工作目录（存储向量数据库和知识图谱）
WORKING_DIR=data/rag_storage

# 文档库目录（用于 Gradio 界面的文档管理）
DOC_LIBRARY=data/inputs
```

这些配置会影响：
- `insurance_rag_gradio.py` - Gradio Web 界面
- `RAGAgent.create()` - Agent 初始化时的工作目录

## ⚙️ LightRAG 系统参数配置

以下是 LightRAG 系统的高级参数配置。这些参数已在 `constants.py` 中设置了默认值，您可以在 `.env` 文件中覆盖这些默认值。

### 服务器配置

```bash
# Gunicorn 工作进程数（用于 API 服务器）
WORKERS=2

# Gunicorn 服务超时时间（秒）
TIMEOUT=300

# 知识图谱允许的最大节点数（防止图谱过大）
MAX_GRAPH_NODES=1000
```

### 信息抽取配置

控制从文档中抽取实体和关系的行为：

```bash
# 文档处理默认使用的语言
SUMMARY_LANGUAGE=Chinese

# 默认最多梳理轮次（实体抽取时的迭代次数，值越大越精确但越慢）
MAX_GLEANING=1

# 实体类型列表（JSON数组格式，定义要抽取哪些类型的实体）
ENTITY_TYPES=["Person", "Creature", "Organization", "Location", "Event", "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"]
```

**注意**：`ENTITY_TYPES` 必须是合法的 JSON 数组格式。

### LLM 总结配置

控制何时触发 LLM 对实体/关系描述进行总结：

```bash
# 描述片段数量达到该值则触发 LLM 总结（避免描述过于冗长）
FORCE_LLM_SUMMARY_ON_MERGE=8

# 触发 LLM 总结时允许的最大描述令牌数量
SUMMARY_MAX_TOKENS=1200

# 推荐的 LLM 输出摘要字数（单位：token）
SUMMARY_LENGTH_RECOMMENDED=600

# 总结时传给 LLM 的最大上下文 token 数
SUMMARY_CONTEXT_SIZE=12000
```

### 查询和检索配置

调整检索行为和上下文大小：

```bash
# 全局召回数（向量检索返回的最大结果数）
TOP_K=40

# 文档片段召回 top-K
CHUNK_TOP_K=20

# 实体最大 token 数（上下文中实体描述的最大 token）
MAX_ENTITY_TOKENS=6000

# 关系描述最大 token 数（上下文中关系描述的最大 token）
MAX_RELATION_TOKENS=8000

# LLM 单轮最大可处理 token 数（包括系统提示、实体、关系和文档片段）
MAX_TOTAL_TOKENS=30000

# 向量检索相似度阈值（cosine similarity，0-1之间）
COSINE_THRESHOLD=0.2

# 相关文档片段默认数量（从单个实体或关系获取的相关片段数）
RELATED_CHUNK_NUMBER=5

# 知识片段选择方法（VECTOR: 基于向量相似度, WEIGHT: 基于权重）
KG_CHUNK_PICK_METHOD=VECTOR
```

**调优建议**：
- 如果回答不够详细，可以增加 `MAX_ENTITY_TOKENS`、`MAX_RELATION_TOKENS` 或 `MAX_TOTAL_TOKENS`
- 如果检索到的内容不够相关，可以提高 `COSINE_THRESHOLD`（但不要超过 0.5）
- `KG_CHUNK_PICK_METHOD=VECTOR` 通常效果更好，但 `WEIGHT` 可能在某些场景下更快

### 重排序配置

```bash
# 最小 rerank 分数（精排后过滤片段的最小分数阈值）
MIN_RERANK_SCORE=0.0
```

设置更高的值可以过滤掉不相关的结果，但可能会丢失一些有用信息。

### 异步处理配置

控制系统的并发度：

```bash
# 最大异步执行数量（LLM 调用的并发数）
MAX_ASYNC=4

# 最大并发插入操作数（向向量数据库插入数据时的并发数）
MAX_PARALLEL_INSERT=2
```

**性能调优**：
- 如果 API 限流，减小 `MAX_ASYNC`
- 如果有充足的 API 配额，可以增加 `MAX_ASYNC` 提高速度
- 向量数据库性能好时，可以增加 `MAX_PARALLEL_INSERT`

### 向量嵌入配置

```bash
# 嵌入计算最大异步数量
EMBEDDING_FUNC_MAX_ASYNC=8

# 嵌入批处理默认数量
EMBEDDING_BATCH_NUM=10
```

这些参数影响 Embedding 计算的并发度和批处理大小。

### 超时配置

```bash
# LLM 超时时间（秒）
LLM_TIMEOUT=180

# Embedding 超时时间（秒）
EMBEDDING_TIMEOUT=30
```

如果使用较慢的模型或网络不稳定，可以适当增加超时时间。

### 日志配置

```bash
# 日志文件最大字节数（默认 10MB）
LOG_MAX_BYTES=10485760

# 保留日志备份数量
LOG_BACKUP_COUNT=5

# 默认日志文件名
LOG_FILENAME=lightrag.log
```

## 🔄 参数优先级说明

系统使用以下优先级顺序来确定配置值：

1. **✅ 最高优先级：`.env` 环境变量**
   - 在 `.env` 文件中设置的任何参数都会覆盖默认值
   - 示例：`TOP_K=50` 会覆盖 `constants.py` 中的 `DEFAULT_TOP_K=40`

2. **📦 默认优先级：`constants.py` 中的常量**
   - 如果 `.env` 中没有设置某个参数，系统会使用 `constants.py` 中定义的默认值
   - 这些默认值经过调优，适合大多数场景

**工作原理**：

代码通过 `get_env_value()` 函数读取配置：

```python
from .utils import get_env_value
from .constants import DEFAULT_TOP_K

# 优先读取环境变量，如果没有则使用 constants.py 中的默认值
top_k = get_env_value("TOP_K", DEFAULT_TOP_K, int)
```

**最佳实践**：
- 保持 `constants.py` 不变，它提供了合理的默认值
- 仅在 `.env` 中覆盖需要自定义的参数
- 这样可以在不修改代码的情况下灵活调整配置

### 参数优先级示例

假设 `constants.py` 中有以下默认值：

```python
# constants.py
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20
DEFAULT_MAX_ENTITY_TOKENS = 6000
```

**场景 1：不设置环境变量**

`.env` 文件中不包含这些参数，系统使用默认值：
- `TOP_K` = 40 （来自 `constants.py`）
- `CHUNK_TOP_K` = 20 （来自 `constants.py`）
- `MAX_ENTITY_TOKENS` = 6000 （来自 `constants.py`）

**场景 2：部分覆盖**

`.env` 文件中设置：
```bash
TOP_K=60
MAX_ENTITY_TOKENS=8000
```

最终结果：
- `TOP_K` = 60 ✅ （`.env` 覆盖）
- `CHUNK_TOP_K` = 20 （来自 `constants.py`）
- `MAX_ENTITY_TOKENS` = 8000 ✅ （`.env` 覆盖）

**场景 3：完全覆盖**

`.env` 文件中设置：
```bash
TOP_K=100
CHUNK_TOP_K=50
MAX_ENTITY_TOKENS=10000
```

最终结果：
- `TOP_K` = 100 ✅ （`.env` 覆盖）
- `CHUNK_TOP_K` = 50 ✅ （`.env` 覆盖）
- `MAX_ENTITY_TOKENS` = 10000 ✅ （`.env` 覆盖）

## 🎯 最佳实践

1. **安全性**: 永远不要将 `.env` 文件提交到 Git 仓库（已在 `.gitignore` 中）
2. **本地开发**: 使用 Ollama 作为 Embedding，速度快且完全本地化
3. **生产环境**: 考虑使用云端 Embedding API 或部署自己的 vLLM 服务
4. **精排模型**: 对于精度要求高的场景，建议启用 Rerank（会增加推理时间）
5. **GPU 配置**: 如果有多张 GPU，可以通过 `RERANK_DEVICE=cuda:1` 分配到不同的 GPU

## 🐛 故障排查

### 问题 1: 提示找不到 API Key

**解决方案**: 检查 `.env` 文件是否在项目根目录，且相应的 API Key 已正确填写。

### 问题 2: Ollama 连接失败

**解决方案**: 
1. 确保 Ollama 服务正在运行: `ollama serve`
2. 检查 `OLLAMA_BASE_URL` 是否正确
3. 确认模型已下载: `ollama pull qwen3-embedding:0.6b`

### 问题 3: Reranker 模型加载失败

**解决方案**:
1. 如果使用本地路径，确保路径存在且包含模型文件
2. 如果从 HuggingFace 下载，确保网络连接正常
3. 检查是否有足够的磁盘空间和内存

### 问题 4: Embedding 维度不匹配

**解决方案**: 确保 `EMBEDDING_DIM` 与实际模型的输出维度匹配：
- `qwen3-embedding:0.6b`: 1024
- `bge-m3`: 1024
- `text-embedding-3-large`: 1536
- `all-MiniLM-L6-v2`: 384

## 📚 相关文档

- [环境变量模板](env_template.txt)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 🤝 贡献

如果你发现任何问题或有改进建议，欢迎提交 Issue 或 Pull Request。

