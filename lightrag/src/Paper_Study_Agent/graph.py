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
from .lightrag_core import LightRAGKnowledgeGraph

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
     "用户问题：{query}\n\n"
     "对话历史检索：\n{history_retrieved}\n\n"
     "传统文档检索：\n{context_retrieved}\n\n"
     "知识图谱增强检索：\n{graph_context}\n\n"
     "请优先使用知识图谱信息，结合传统检索内容，用对话式语气回复。"
     "如果知识图谱提供了更相关的信息，请重点参考。"
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
    use_lightrag = state.get("use_lightrag", False)

    print(f"正在为 {len(docs)} 个文档块生成嵌入...")
    main_vstore = FAISS.from_documents(docs, embedder)
    convstore = default_FAISS(embedder)

    print(f"✅ 向量库构建完成，共 {main_vstore.index.ntotal} 个向量")
    
    # 构建知识图谱（如果启用 LightRAG）
    knowledge_graph = None
    if use_lightrag:
        print("🧠 开始构建 LightRAG 知识图谱...")
        knowledge_graph = LightRAGKnowledgeGraph(embedder)
        graph_result = knowledge_graph.build_graph(docs)
        print(f"✅ 知识图谱构建完成: {graph_result['graph_stats']}")

    return Command(
        goto="retrieve",
        update={
            "vectorstore": main_vstore,
            "convstore": convstore,
            "knowledge_graph": knowledge_graph
        }
    )

def retrieve(state: Paper_Study_State) -> Command:
    question = state["query"]
    vectorstore = state["vectorstore"]
    convstore = state["convstore"]
    knowledge_graph = state.get("knowledge_graph")

    # 传统向量检索
    docs_context = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
    reordered_context = long_reorder.transform_documents(docs_context)
    context_str = docs2str(reordered_context)

    # LightRAG 图增强检索
    graph_context = ""
    if knowledge_graph:
        print("🔍 使用 LightRAG 进行图增强检索...")
        graph_docs = knowledge_graph.graph_enhanced_retrieve(question, k=3)
        if graph_docs:
            graph_context = docs2str(graph_docs)
            print(f"✅ 从知识图谱检索到 {len(graph_docs)} 个相关文档")
        else:
            print("⚠️ 知识图谱未检索到相关文档")

    # 对话历史检索
    docs_history = convstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    reordered_history = long_reorder.transform_documents(docs_history)
    history_str = docs2str(reordered_history) if docs_history else "无相关对话历史"

    return Command(
        goto="generate_answer",
        update={
            "context_retrieved": context_str,
            "graph_context": graph_context,
            "history_retrieved": history_str
        }
    )

def generate_answer(state: Paper_Study_State) -> Command:
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state["query"],
        "context_retrieved": state["context_retrieved"],
        "graph_context": state.get("graph_context", ""),
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