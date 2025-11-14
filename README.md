# 🏦 知识图谱问答助手 (Knowledge Graph Agent)
## 🗒️ 更新日志
- **2025.10.02** @masterlyj
  建立项目，初始化架构
- **2025.10.07** @masterlyj
  添加嵌入模型，统一模型导入接口
- **2025.10.08** @ziyuchen-cyber
  适配 deepseek 模型，扩展 llm 导入接口
- **2025.10.10** @masterlyj
  比赛 agent 正式搭建，支持前期实体关系抽取
- **2025.10.11** @xinyi-kinley
  修改了prompt 提示词，润色实体关系抽取模板
- **2025.10.12** @masterlyj
  完成 agent 全流程搭建，前端设计以gradio为基础
- **2025.10.14** @masterlyj
  修复bug，解决无法进行多轮对话（崩溃）问题，提升上下文理解与交互连贯性。
- **2025.10.15** @jude  
  修复 LightRAG 与 DeepSeek 兼容性问题，集成 Mineru 到 RAG 工作流，实现网页端 PDF 上传能力。
- **2025.10.21** @masterlyj
  删除了无用代码，修复了 mineru 解析无法下载 md 的 "pending" bug
- **2025.10.22** @sunny1dan
  上传比赛所需要的所有解析 md 文件
- **2025.10.23** 
  检索粗排时考虑时效性，增添答案时效性能力；@jude
  为检索粗排的文本块添加rerank模型，打分体系已构建；@ziyuchen-cyber
- **2025.10.27** @masterlyj
  修复检索文本块不全的问题，对整体流程进行了细节优化。
- **2025.10.28** @masterlyj
  实现了检索子图的知识图谱可视化，
- **2025.10.29** @ziyuchen-cyber
  知识图谱细节展示优化，环境变量配置统一管理。
- **2025.10.30**
  实现 gradio 共享链接，公网可访问。@masterlyj
  支持postgresql 数据库，实现数据持久化存储。@jude
- **2025.11.02** @masterlyj
  知识图谱节点颜色多样化（细节优化）；
  多轮对话记忆轮次实现；
  设置数据库存储模式（内存+数据库）；
  前端js,css,html分离管理。
- **2025.11.03** @masterlyj
  top-k参数优先级优化；
  文本块添加废止时间字段且可被索引。
- **2025.11.06** 
  知识图谱提示词路由优化；@masterlyj
  流式输出；
  输出推理过程；
  前段界面优化。@ziyuchen-cyber
- **2025.11.11** @masterlyj
  修改深度思考模式，；
  配置可缩放；
  支持vllm框架的模型接入。
- **2025.11.13** @masterlyj
  deepseek-chat与deepseek-reasoner路由适配；
  深度思考提示词修改；
  修改query关键词提取秒切deepseek-chat。

## 🚧 后续优化方向（TODO）
- [ ] 领域增强与两级模型协同
  [ ] 数据型表格入库与工具化查询
  [ ] 引入迭代探索裁剪推理


## ✨ 项目简介

知识图谱问答助手是一个基于 GraphRAG 和 LangGraph 构建的智能 RAG 检索系统，专为处理保险领域的复杂文档而设计。它能够从多类型保险文档中提取实体和关系，构建知识图谱，并结合向量检索、关键词检索和图谱推理，提供精准、时效性强的答案。该助手通过 Gradio 提供直观的用户界面，支持多种查询模式，适用于寿险条款、产品说明书、理赔指南等保险文档。

## 🚀 核心功能

*   **多文档智能索引**: 支持批量上传和索引 `md`, `txt`, `pdf` 等格式的保险文档。
*   **混合检索**: 融合向量相似度匹配和知识图谱推理，提供更准确、全面的检索结果。
*   **多种查询模式**:
    *   **综合检索 (mix, 推荐)**: 同时利用知识图谱数据与向量检索到的文档片段，覆盖面最全，默认模式。
    *   **混合检索 (hybrid)**: 融合本地实体视角（local）与全局关系视角（global）的知识图谱结果（轮询合并），不引入纯向量片段。
    *   **向量检索 (naive)**: 纯语义相似度匹配，速度快。
    *   **局部图谱 (local)**: 基于实体关系的邻域搜索。
    *   **全局图谱 (global)**: 全图推理，适合复杂关联查询。
*   **知识图谱构建**: 从非结构化文本中抽取实体、关系，并构建统一的知识图谱。
*   **对话记忆**: 维护对话历史，提供上下文感知的交互。
*   **直观 Gradio 用户界面**: 提供用户友好的 Web 界面，支持文档管理、查询和结果展示。
*   **LightRAG 框架**: 利用 LightRAG 提供的异步能力和模块化设计，实现高效的 RAG 工作流。

## 🏗️ 架构概览 (LangGraph 工作流)

知识图谱问答助手的工作流通过 LangGraph 精心编排，主要包含两个独立但协同的流程：文档索引工作流和查询工作流。

### 1. 文档索引工作流 (`create_indexing_graph`)

这是一个简化的工作流，主要负责将原始文档导入到 LightRAG 系统中进行处理。

*   **`index_documents` 节点**：
    *   **职责**: 接收待索引的文档内容和元数据 (文件路径、ID)，触发 LightRAG 内部的完整索引流程。LightRAG 将负责文档切分、Embedding、实体抽取、关系构建，并将数据存储到向量数据库和知识图谱中。
    *   **输出**: 索引任务的跟踪 ID 和状态消息。

### 2. 查询工作流 (`create_querying_graph`)

此工作流负责处理用户的问题，并生成基于知识图谱和文档内容的回答。

*   **`retrieve_context` 节点**：
    *   **职责**: 基于用户查询和选择的查询模式 (mix, hybrid, naive, local, global)，调用 LightRAG 的查询接口，从存储中检索最相关的上下文信息（包括文档片段和知识图谱数据）。
    *   **输出**: 原始上下文 (`raw_context`) 和查询模式 (`query_mode`)。
*   **`generate_answer` 节点**：
    *   **职责**: 接收检索到的上下文和用户查询，将其作为 Prompt 传递给 LLM，生成最终的专业回答。
    *   **输出**: LLM 生成的答案 (`answer`)。

## ⚙️ 快速开始

### 1. 环境准备

确保您的系统已安装 Python 3.11+。

### 2. 克隆项目

```bash
git clone https://github.com/yourusername/AI_Agent.git # 请替换为您的项目实际地址
cd AI_Agent
```

### 3. 安装依赖

本项目推荐使用 `uv` 进行依赖管理，以获得更快的安装和更可靠的依赖解析。

```bash
# 如果尚未安装 uv，请先安装
pip install uv

# 创建并激活虚拟环境
uv venv
source .venv/bin/activate # macOS/Linux
# 或 .venv\Scripts\activate # Windows

# 安装项目依赖并同步到 lock 文件
uv sync
```

如果您习惯使用 `pip`：

```bash
python -m venv .venv
source .venv/bin/activate # 或 .venv\Scripts\activate
pip install -e .
```

### 4. 配置 Ollama (Embedding 模型)

本项目默认使用 Ollama 作为 Embedding 服务。

*   **安装 Ollama**：请访问 [ollama.ai](https://ollama.ai/)，按照官方指南安装并启动 Ollama 服务。
*   **下载 Embedding 模型**：在终端中运行以下命令下载 `qwen3-embedding:0.6b` 模型：

    ```bash
    ollama run qwen3-embedding:0.6b
    # 等待模型下载完成。Ollama 会自动在后台运行，并监听 http://localhost:11434
    ```

    请确保 Ollama 服务在 `http://localhost:11434` 端口（或您配置的任何端口）正常运行。

### 5. PostgreSQL数据库安装(可选)

本项目默认使用 内存（memory） 存储。如果您需要使用 PostgreSQL 存储，请按照以下步骤配置。（完成下述配置后，可在界面左侧「💾 存储配置」面板切换为“数据库存储”；如需默认即为数据库存储，可在 `src/Knowledge_Graph_Agent/insurance_rag_gradio.py` 将 `current_storage_mode` 初始值设置为 `"database"`）

*   **备注**：windows用户建议在linux虚拟机中安装PostgreSQL，否则age扩展无法安装。mac用户应该可以在本地安装PostgreSQL（未测试过）。
本系统PostgreSQL数据库详细配置请参考`env_template.txt`文件中的注释。

*   **安装 PostgreSQL(需要安装16.6以上的版本)**：请访问 [postgresql.org](https://www.postgresql.org/)，按照官方指南安装并启动 PostgreSQL 服务。(可以选择安装pgAdmin 4工具方便管理数据库)

*   **安装 pgvector 扩展**：在 PostgreSQL 中安装 `pgvector` 扩展，用于存储和查询向量数据。请参考 [pgvector 官方文档](https://github.com/pgvector/pgvector) 进行安装。

*   **安装 age 扩展**：在 PostgreSQL 中安装 `age` 扩展，用于存储和查询知识图谱数据。请参考 [age 官方文档](https://github.com/apache/age/tree/master) 进行安装。

*   **成功安装上述扩展后需要进入数据库并创建扩展**：
    ```sql
    -- 进入 PostgreSQL 数据库
    psql -U postgres -d postgres

    -- 创建 pgvector 扩展
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 创建 age 扩展
    CREATE EXTENSION IF NOT EXISTS age;
    ```




### 6. 项目配置（强烈建议从 env 模板拷贝）

先创建并编辑 `.env`：

```bash
cp env_template.txt .env
```

以下为最小可运行配置示例（按需替换为你的 Key/模型）。更多可选项请查看 `ENV_CONFIG_GUIDE.md` 与 `env_template.txt` 注释。

```bash
# ---- LLM（两选一，按你使用的服务填写）----
LLM_PROVIDER=google_genai              # 可选: google_genai | deepseek
LLM_MODEL=gemini-2.5-flash             # DeepSeek 可用 deepseek-chat
GOOGLE_API_KEY=你的_Google_API_Key     # 若用 DeepSeek 则改填 DEEPSEEK_API_KEY
# DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
# LLM_TEMPERATURE=0

# ---- Embedding（默认使用本地 Ollama）----
EMBEDDING_TYPE=ollama
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_DIM=1024

# ---- 工作目录与文档目录 ----
WORKING_DIR=data/rag_storage
DOC_LIBRARY=data/inputs

# ---- 可选：开启精排 Rerank（默认关闭）----
# RERANK_ENABLED=true
# RERANK_MODEL=maidalun1020/bce-reranker-base_v1
# RERANK_TOP_K=20

# ---- 可选：PDF 解析（MinerU）----
# MINERU_API_KEY=你的_MinerU_API_Key
```

要点：
- LLM 支持 Google Gemini 与 DeepSeek。未显式设置 `LLM_PROVIDER` 时，会按可用 Key 自动回退。
- Embedding 支持 HuggingFace、Ollama、vLLM。默认已配置 Ollama，本地即可运行。
- 如需数据库存储请填写 `env_template.txt` 中的 PostgreSQL 段，并在界面切换“数据库存储”。

### 7. 准备文档

将您需要索引的保险文档（如寿险条款、产品说明书、理赔指南等，支持 `.md`）放置在 `data/inputs` 目录下。

说明：
- 支持在界面直接上传 `PDF/MD/TXT`。若配置了 `MINERU_API_KEY`，PDF 会自动解析为 Markdown 再入库；否则仅索引 MD/TXT。
- 初次运行会在 `data/rag_storage` 生成本地存储（内存模式）。数据库模式下数据写入 PostgreSQL。

### 8. 运行应用

在激活的虚拟环境中，执行以下命令启动 Gradio Web UI：

```bash
python -m src.Knowledge_Graph_Agent.insurance_rag_gradio
```

程序启动后，您会在终端看到一个本地 URL (例如 `http://127.0.0.1:7860`)。在浏览器中打开此链接，即可开始与知识图谱问答助手进行交互！

运行小贴士：
- 左侧「📁 文档库管理」可上传并索引文档，索引统计会实时反馈。
- 左侧「💾 存储配置」可在“内存管理/数据库存储”之间切换，点击“应用存储模式”后系统会重新初始化。
- 顶部输入框支持多种查询模式（mix/hybrid/naive/local/global），可选开启 Rerank 精排与上下文展示。

## 💡 示例交互

启动应用后，您可以尝试提出以下问题：

*   "什么情况下保险公司会豁免保险费?"
*   "犹豫期是多长时间?解除合同有什么后果?"
*   "全残的定义包括哪些情况?"
*   "保险责任和责任免除有什么区别?"
*   "投保人年龄错误会如何处理?"

