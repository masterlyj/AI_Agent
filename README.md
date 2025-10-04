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

## 💡 示例交互

启动应用后，您可以尝试提出以下问题：

*   "Graph RAG 是什么？"
*   "这篇论文提出了哪些创新点？"
*   "能否总结一下这篇论文的核心结论？"