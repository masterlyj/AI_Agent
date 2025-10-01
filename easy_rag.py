from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import bs4

# 1. 加载文档
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs_list = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=400
)
doc_splits = text_splitter.split_documents(docs_list)

# 3. 创建向量库并初始化检索器
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, 
    embedding = OpenAIEmbeddings(
        model="Qwen3-Embedding-0.6B",      
        base_url="http://localhost:18889/v1",
        api_key="not-needed",
    )
)

retriever = vectorstore.as_retriever()

# 4. 创建检索工具
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retrieve_blog_posts",
    description="搜索并返回有关莉莲·翁（Lilian Weng）博客文章的信息。",
)

llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    api_key="AIzaSyA3ESlDpHlLq-rQJa_ycsrdtUwYBm1UpEc",
)

# 5. 决策节点：决定是检索还是直接回答
def generate_query_or_respond(state: MessagesState):
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

# 6. 回答生成节点：基于检索到的上下文生成答案
def generate_answer(state: MessagesState):
    # 提取原始问题和检索到的上下文
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = (
        "你是一个用于问答任务的助手。"
        "请**严格根据以下检索到的上下文**来回答问题。"
        "**如果原文对某概念有明确分类，请务必在回答中清晰地列出这些类别。**"
        "如果你不知道答案，只需说你不知道。"
        "使用详尽的内容来回答，通俗易懂，并给出具体原文内容引用\n"
        f"问题：{question}\n上下文：{context}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# 7. 构建工作流图
workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("生成查询或回应", generate_query_or_respond)
workflow.add_node("检索", ToolNode([retriever_tool])) # 自动执行工具
workflow.add_node("生成答案", generate_answer)

# 设置入口点
workflow.add_edge(START, "生成查询或回应")

# 从决策节点出发的条件边：如果模型决定调用工具，则去 retrieve；否则直接结束（END）
workflow.add_conditional_edges(
    "生成查询或回应",
    tools_condition,
    {"tools": "检索", END: END},
)

# 从 retrieve 节点直接到 generate_answer
workflow.add_edge("检索", "生成答案")
workflow.add_edge("生成答案", END)

# 编译成可执行的图
graph = workflow.compile()

# 提问
user_input = "奖励黑客是什么？请说出莉莲·翁（Lilian Weng）对奖励黑客的类型有何看法？"

# 流式输出每一步的结果
for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
    for node_name, node_data in event.items():
        print(f"--- Node: {node_name} ---")
        node_data["messages"][-1].pretty_print()
        print("\n")