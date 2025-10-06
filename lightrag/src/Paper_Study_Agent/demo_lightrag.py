"""
LightRAG 集成演示
展示如何在 LangGraph 框架中使用 LightRAG 进行增强的论文问答
"""

import gradio as gr
from .app import PaperChatBot

# 配置
ARXIV_IDS = ["2410.05779", "2404.16130"]  # LightRAG 论文和其他相关论文
EMBEDDING_CONFIG = {
    "type": "ollama",
    "model": "qwen3-embedding:0.6b"
}

# 创建带 LightRAG 的聊天机器人
bot_with_lightrag = PaperChatBot(
    arxiv_ids=ARXIV_IDS, 
    embedding_config=EMBEDDING_CONFIG, 
    use_lightrag=True
)

# 创建传统 RAG 的聊天机器人（用于对比）
bot_traditional = PaperChatBot(
    arxiv_ids=ARXIV_IDS, 
    embedding_config=EMBEDDING_CONFIG, 
    use_lightrag=False
)

def compare_responses(message, history):
    """对比传统 RAG 和 LightRAG 的回答"""
    
    # 获取 LightRAG 回答
    lightrag_response = bot_with_lightrag.chat(message, history)
    
    # 获取传统 RAG 回答
    traditional_response = bot_traditional.chat(message, history)
    
    # 格式化对比结果
    comparison = f"""
## 🔍 回答对比

### 🧠 LightRAG 增强回答
{lightrag_response}

---

### 📚 传统 RAG 回答  
{traditional_response}

---
*LightRAG 通过知识图谱提供更丰富的上下文关联和更准确的答案*
"""
    
    return comparison

# 创建对比界面
with gr.Blocks(title="📚 LightRAG vs 传统 RAG 对比") as demo:
    gr.Markdown("# 🚀 LightRAG 集成演示")
    gr.Markdown("""
    这个演示展示了 LightRAG 与传统 RAG 在论文问答任务中的对比。
    
    **LightRAG 的优势：**
    - 🧠 知识图谱增强检索
    - 🔗 实体关系理解
    - 📈 更准确的上下文关联
    - 🎯 更精准的答案生成
    """)
    
    with gr.Row():
        with gr.Column():
            message_input = gr.Textbox(
                label="💬 输入您的问题",
                placeholder="例如：LightRAG 相比传统 RAG 有什么优势？",
                lines=2
            )
            
            submit_btn = gr.Button("🚀 对比回答", variant="primary")
        
        with gr.Column():
            response_output = gr.Markdown(
                label="📊 回答对比",
                value="请输入问题开始对比..."
            )
    
    # 示例问题
    gr.Examples(
        examples=[
            "LightRAG 的核心创新点是什么？",
            "知识图谱在 RAG 中的作用是什么？",
            "LightRAG 相比 GraphRAG 有什么优势？",
            "这个系统如何处理实体关系？",
            "LightRAG 的检索策略有什么特点？"
        ],
        inputs=message_input
    )
    
    # 事件绑定
    submit_btn.click(
        fn=compare_responses,
        inputs=[message_input, gr.State([])],
        outputs=response_output
    )
    
    message_input.submit(
        fn=compare_responses,
        inputs=[message_input, gr.State([])],
        outputs=response_output
    )

# 单独使用 LightRAG 的界面
lightrag_only_demo = gr.ChatInterface(
    fn=bot_with_lightrag.chat,
    description=bot_with_lightrag.get_initial_message(),
    title="🧠 LightRAG 论文问答助手",
    examples=[
        "LightRAG 的核心创新点是什么？",
        "知识图谱如何提升检索效果？", 
        "LightRAG 与传统 RAG 的区别？"
    ],
    cache_examples=False,
    type="messages"
).queue()

if __name__ == "__main__":
    print("🚀 启动 LightRAG 演示...")
    print("📊 对比界面: http://127.0.0.1:7860")
    print("🧠 LightRAG 专用界面: http://127.0.0.1:7861")
    
    # 启动对比界面
    demo.launch(server_port=7860, share=False)
    
    # 启动 LightRAG 专用界面（在另一个端口）
    # lightrag_only_demo.launch(server_port=7861, share=False)

