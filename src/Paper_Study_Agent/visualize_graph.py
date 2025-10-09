from typing import List
from langgraph.graph import StateGraph, END

from .app import PaperChatBot

def generate_and_save_graph(graph_obj: StateGraph, filename: str):
    """
    生成 LangGraph 工作流图的可视化，并保存为 PNG 文件。
    """
    try:
        print(f"正在生成工作流图 {filename}...")
        png_data = graph_obj.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2)

        # 将 PNG 数据写入文件
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"图形可视化已保存至 {filename}")

    except Exception as e:
        print(f"绘制图形时出错: {e}")

if __name__ == "__main__":
    ARXIV_IDS = ["2410.05779", "2404.16130"]
    EMBEDDING_CONFIG = {
        "type": "ollama",
        "model": "qwen3-embedding:0.6b"
    }

    print("正在初始化 PaperChatBot 以获取图对象...")
    try:
        # 实例化 PaperChatBot 会运行其初始化逻辑，包括论文加载和向量化
        chatbot_instance = PaperChatBot(arxiv_ids=ARXIV_IDS, embedding_config=EMBEDDING_CONFIG)
        
        # 获取要可视化的编译后的图
        paper_study_workflow_graph = chatbot_instance.graph

        graphs_to_generate = {
            paper_study_workflow_graph: "paper_study_workflow_graph.png",
        }

        for graph_obj, filename in graphs_to_generate.items():
            generate_and_save_graph(graph_obj, filename)
            
    except Exception as e:
        print(f"初始化 PaperChatBot 或生成图表时发生错误: {e}")
        print("请确保网络连接正常，arxiv_ids 有效，并且嵌入模型配置正确。")