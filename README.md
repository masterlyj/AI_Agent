# 📚 论文问答助手 (Paper Study Agent)

## ✨ 项目简介

本项目旨在一个智能化的框架，帮助用户高效地与学术论文进行交互和学习。它结合了最新的检索增强生成 (RAG) 技术与 LangGraph 强大的状态管理能力，能够从 arXiv 加载论文，并利用本地部署的 Ollama Embedding 模型进行向量化，最终通过大型语言模型 (LLM) 提供精准、连贯且有据可查的回答。无论是快速概览论文核心思想，还是深入探讨特定细节，本助手都将是您可靠的学术伙伴。

## 🚀 核心功能

*   **arXiv 论文智能加载**：通过提供 arXiv ID，系统能够自动获取并处理目标学术论文。
*   **高效文档分块与嵌入**：采用先进的文本分块策略，并结合 **Ollama Embedding 模型 (如 `qwen3-embedding:0.6b`)** 将论文内容转化为高质量的向量表示，为精准检索奠定基础。
*   **LangGraph 驱动的 RAG 架构**：利用 LangGraph 构建灵活、可追溯的 RAG 工作流，确保问答过程的逻辑清晰和结果的准确性。
*   **智能对话记忆**：系统能够感知并利用之前的对话历史，提供更自然、上下文感知的交互体验。
*   **直观 Gradio 用户界面**：提供用户友好的 Web 界面，让您轻松上传论文、提问并获取答案。

## 🏗️ 架构概览 (LangGraph 工作流)

本项目的核心是一个精心设计的 LangGraph 状态图，其主要节点及职责如下：

1.  **`load_and_chunk_papers`**：
    *   **职责**：从 arXiv 加载指定论文，并运用 `RecursiveCharacterTextSplitter` 对其内容进行优化分块。
    *   **输出**：处理后的文档块列表 (`context`)。
2.  **`embed_and_index`**：
    *   **职责**：接收文档块，使用配置的 Embedding 模型（例如 Ollama `qwen3-embedding:0.6b`）生成向量嵌入，并将文档块及其向量存储到 FAISS 向量数据库中，同时初始化对话历史向量库。
    *   **输出**：论文内容向量库 (`vectorstore`) 和对话历史向量库 (`convstore`)。
3.  **`retrieve`**：
    *   **职责**：基于用户当前查询，并行地从 `vectorstore` (论文内容) 和 `convstore` (对话历史) 中检索最相关的上下文信息。
    *   **策略**：采用 `LongContextReorder` 对检索结果进行重排序，以优化 LLM 处理长上下文时的性能。
    *   **输出**：检索到的论文上下文 (`context_retrieved`) 和对话历史 (`history_retrieved`)。
4.  **`generate_answer`**：
    *   **职责**：将用户查询与检索到的高质量上下文（论文内容和对话历史）结合，作为 Prompt 传递给 LLM，生成最终的回答。
    *   **输出**：LLM 生成的回答 (`answer`)。
5.  **`update_convstore`**：
    *   **职责**：将当前轮次的用户查询和 LLM 回答添加到 `convstore` 中，更新对话记忆，为后续交互提供更丰富的上下文。
    *   **输出**：更新后的对话历史向量库 (`convstore`)。

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

### 5. 项目配置

#### 5.1 Embedding 模型配置

打开 `src/Paper_Study_Agent/demo.py` 文件，确保 `EMBEDDING_CONFIG` 已正确指向 Ollama 模型：

```python
# src/Paper_Study_Agent/demo.py
# ...
EMBEDDING_CONFIG = {
    "type": "ollama",
    "model": "qwen3-embedding:0.6b",
    # 如果您的 Ollama 服务运行在非默认地址或端口，请取消注释并修改 base_url
    # "base_url": "http://your-ollama-host:port/v1"
}
# ...
```

#### 5.2 LLM 配置 (可选)

您可以在 `src/Paper_Study_Agent/llm.py` 中配置您希望使用的大型语言模型。默认配置可能已为您准备好，但您可以根据需要更改为 OpenAI、Mistral 或其他 Ollama 兼容的 LLM。

### 6. 运行应用

在激活的虚拟环境中，执行以下命令启动 Gradio Web UI：

```bash
python -m src.Paper_Study_Agent.demo
```

程序启动后，您会在终端看到一个本地 URL (例如 `http://127.0.0.1:7860`)。在浏览器中打开此链接，即可开始与论文问答助手进行交互！

## 🚀 项目规划与任务分工

### 1. 实体检测与词表构建 (1人负责)

*   **设计**：分层分类的保险术语本体，包含专业术语与通俗表达的映射关系。
*   **实施**：通过提示词工程，让大模型从保险文档中提取标准化实体，建立业务概念词典。
*   **产出**：
    *   保险领域实体分类体系
    *   专业-通俗术语映射表
    *   实体抽取提示词模板

### 2. 用户意图理解与问题重写 (0人负责)

*   **设计**：利用构建的保险术语词典，将用户口语化问题重写为标准化专业查询。
*   **实施**：设计意图识别提示词，通过术语映射实现问题规范化。
*   **验证**：通过反推法从知识图谱实体生成测试用例，构建真实销售场景对话数据集。
*   **产出**：
    *   意图分类体系
    *   问题重写提示词
    *   语义表达测试数据集

### 3. 知识图谱构建与多文档融合 (4人负责)

*   **方法**：通过提示词从多类型保险文档中抽取实体关系，构建统一知识图谱。可以参考 `src/Knowledge_Graph_Agent/prompt.py` 中定义提示词的模式，以及 `src/Knowledge_Graph_Agent/utils_graph.py` 和 `src/Knowledge_Graph_Agent/kg` 子目录下的图谱操作和存储实现。
*   **实施**：设计针对费率表、产品说明、合同等不同文档类型的抽取提示词。
*   **存储**：利用9张PostgreSQL表实现向量化存储和图结构存储的协同。`src/Knowledge_Graph_Agent/kg/networkx_impl.py` 和 `src/Knowledge_Graph_Agent/kg/nano_vector_db_impl.py` 提供了图和向量存储的思路。
*   **产出**：
    *   知识图谱抽取提示词
    *   多文档融合方案
    *   图谱质量评估标准

### 4. 时效性精准打分系统 (1人负责)

*   **方法**：建立基于时间衰减、监管对齐、产品状态的综合打分机制。
*   **实施**：
    *   为语料添加采集时间、发布时间、有效期等元数据。
    *   设计动态权重规则：新文档奖励、过期文档降权、监管合规加分。
    *   在检索阶段进行时效性重排序。
*   **验证**：确保新文档召回率100%，过期文档误召率<1%。
*   **产出**：
    *   时效性打分算法
    *   元数据规范
    *   重排序策略

### 5. 系统开发与召回优化 (0人负责)

*   **方法**：实现向量检索、关键词检索、图谱检索的三路召回，结合时效性重排序。对 `src/Knowledge_Graph_Agent/light_graph_rag.py` 中检索逻辑的扩展，以及对 `src/Knowledge_Graph_Agent/utils.py` 中相关工具函数的调整。
*   **实施**：
    *   设计多路召回粗排机制。
    *   集成rerank模型进行精排。
    *   结合时效性得分进行最终排序。
*   **产出**：
    *   系统架构设计
    *   召回算法实现
    *   性能优化方案

### 6. 实验设计与效果评估 (0人负责)

*   **方法**：设计基准测试，对比传统RAG与LightRAG改进方案的效果差异。
*   **评估指标**：
    *   ReGAS评价指标、大模型竞争性评价
    *   召回率、准确率等传统指标
    *   时效性相关指标：新文档召回率、过期文档误召率
*   **产出**：
    *   实验设计方案
    *   评估指标体系
    *   对比分析报告

### 7. PDF解析与前端集成 (1人负责)

*   **设计**：集成 mineru 工具，实现 PDF 文档的自动解析、分类与前端展示。
*   **实施**：
    *   调用 mineru API 接口解析 PDF 文档，自动生成 Markdown (md) 文件。
    *   对解析得到的 md 文件进行归类和管理。
    *   开发前端页面，支持文件上传、解析进度展示及解析结果的可视化查看。
*   **产出**：
    *   PDF 解析与归类自动化流程
    *   mineru API 调用与结果处理代码
    *   前端页面（含文件添加与解析结果展示功能）

## 💡 示例交互

启动应用后，您可以尝试提出以下问题：

*   "Graph RAG 是什么？"
*   "这篇论文提出了哪些创新点？"
*   "能否总结一下这篇论文的核心结论？"