import logging

from typing import Dict, List, Any, Annotated, TypedDict, Optional
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faiss import IndexFlatL2
from langgraph.types import Command


from .state import Paper_Study_State
from .llm import get_llm

# --- 初始化 ---
logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

llm = get_llm()
long_reorder = LongContextReorder()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个文档聊天机器人。请根据以下信息回答用户问题。\n"
     "对话历史检索：\n{history_retrieved}\n\n"
     "文档检索：\n{context_retrieved}\n\n"
     "仅根据检索内容回答，用对话式语气回复并给出原文内容（如果原文内容语言与用户问题语言不一致，请将原文内容以及翻译成用于问题语言的内容一起返回）。"
    ),
    ("user", "{query}")
])

# --- 工具函数 ---
def docs2str(docs: List[Document]) -> str:
    return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))

def default_FAISS(embedder) -> FAISS:
    test_vec = embedder.embed_query("test")
    dim = len(test_vec)
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(dim),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

# --- 节点函数 ---
def load_and_chunk_papers(state: Paper_Study_State) -> Command:
    arxiv_ids = state["arXiv_ids"]
    all_chunks: List[Document] = []
    metadata_list = []

    logger.info(f"开始从 arXiv 加载 {len(arxiv_ids)} 篇论文...")

    for arxiv_id in arxiv_ids:
        print(f"正在加载论文: {arxiv_id}")
        try:
            docs = ArxivLoader(query=arxiv_id).load()
            if not docs:
                logger.warning(f"⚠️ 未能加载到文档: {arxiv_id}")
                continue
            doc = docs[0]

            if "References" in doc.page_content:
                doc.page_content = doc.page_content[:doc.page_content.index("References")]

            metadata_list.append(doc.metadata)
            chunks = text_splitter.split_documents([doc])
            filtered_chunks = [c for c in chunks if len(c.page_content) > 200]
            all_chunks.extend(filtered_chunks)

        except Exception as e:
            logger.error(f"❌ 加载论文 {arxiv_id} 时出错: {e}")
            continue

    doc_summary = "可用论文列表：\n"
    for meta in metadata_list:
        title = meta.get("Title", "未知标题")
        doc_summary += f" - {title}\n"

    summary_doc = Document(
        page_content=doc_summary,
        metadata={"source": "paper_summary", "type": "global_context"}
    )
    all_chunks.insert(0, summary_doc)

    logger.info(f"✅ 总共切分块数: {len(all_chunks)}")
    return Command(
        goto="embed_and_index",
        update={"context": all_chunks}
    )

def embed_and_index(state: Paper_Study_State) -> Command:
    embedder = state["embedder"]
    docs = state["context"]

    print(f"正在为 {len(docs)} 个文档块生成嵌入...")
    main_vstore = FAISS.from_documents(docs, embedder)
    convstore = default_FAISS(embedder)

    print(f"✅ 向量库构建完成，共 {main_vstore.index.ntotal} 个向量")
    return Command(
        goto="retrieve",
        update={
            "vectorstore": main_vstore,
            "convstore": convstore
        }
    )

def retrieve(state: Paper_Study_State) -> Command:
    question = state["query"]
    vectorstore = state["vectorstore"]
    convstore = state["convstore"]

    docs_context = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
    reordered_context = long_reorder.transform_documents(docs_context)
    context_str = docs2str(reordered_context)

    docs_history = convstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    reordered_history = long_reorder.transform_documents(docs_history)
    history_str = docs2str(reordered_history) if docs_history else "无相关对话历史"

    return Command(
        goto="generate_answer",
        update={
            "context_retrieved": context_str,
            "history_retrieved": history_str
        }
    )

def generate_answer(state: Paper_Study_State) -> Command:
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state["query"],
        "context_retrieved": state["context_retrieved"],
        "history_retrieved": state["history_retrieved"]
    })
    return Command(
        goto="update_convstore",
        update={"answer": answer}
    )

def update_convstore(state: Paper_Study_State) -> Command:
    convstore = state["convstore"]
    convstore.add_texts([
        f"用户: {state['query']}",
        f"助手: {state['answer']}"
    ])
    return Command(
        goto="__end__",
        update={"convstore": convstore}
    )