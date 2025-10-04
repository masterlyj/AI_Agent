import gradio as gr
from .app import PaperChatBot

# 配置
ARXIV_IDS = ["2410.05779", "2404.16130"]
EMBEDDING_CONFIG = {
    "type": "ollama",
    "model": "qwen3-embedding:0.6b"
}

bot = PaperChatBot(arxiv_ids=ARXIV_IDS, embedding_config=EMBEDDING_CONFIG)

# 启动 Gradio
demo = gr.ChatInterface(
    fn=bot.chat,
    description=bot.get_initial_message(),
    title="📚 论文问答助手",
    examples=["Graph RAG 是什么？", "这篇论文提出了哪些创新？"],
    cache_examples=False,
    type="messages"
).queue()

if __name__ == "__main__":
    demo.launch(debug=True, share=True)