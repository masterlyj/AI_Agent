import gradio as gr
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from .agent import RAGAgent
from .utils import logger

# ===== 全局配置 =====
WORKING_DIR = "data/rag_storage"
DOC_LIBRARY = "data/inputs"

# ===== 自定义CSS样式 =====
custom_css = """
/* 主题色：保险专业蓝 */
:root {
    --primary-color: #1e40af;
    --secondary-color: #3b82f6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --bg-light: #f8fafc;
    --border-color: #e2e8f0;
}

.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

.header-banner {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 30px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.header-banner h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 600;
}

.header-banner p {
    margin: 10px 0 0 0;
    opacity: 0.9;
    font-size: 14px;
}

/* 卡片样式 */
.card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* 聊天消息样式 */
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
}

.assistant-message {
    background: #f1f5f9;
    color: #1e293b;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 85%;
    border-left: 3px solid var(--primary-color);
}

/* 检索指标卡片 */
.metrics-card {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.5);
}

.metric-label {
    font-weight: 500;
    color: #1e40af;
}

.metric-value {
    font-weight: 600;
    color: #1e293b;
}

/* 上下文展示 */
.context-section {
    background: #fefce8;
    border-left: 4px solid var(--warning-color);
    padding: 12px;
    border-radius: 4px;
    margin: 10px 0;
    font-size: 0.9em;
    max-height: 300px;
    overflow-y: auto;
}

/* 实体卡片 */
.entity-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    padding: 4px 12px;
    border-radius: 12px;
    margin: 4px;
    font-size: 0.85em;
}

/* 按钮样式增强 */
.primary-btn {
    background: var(--primary-color) !important;
    color: white !important;
}

.secondary-btn {
    background: white !important;
    color: var(--primary-color) !important;
    border: 1px solid var(--primary-color) !important;
}

/* 状态指示器 */
.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.status-ready { background: var(--success-color); }
.status-indexing { background: var(--warning-color); }
.status-error { background: var(--danger-color); }
"""

# ===== 初始化Agent =====
agent_instance = None
index_status = {"ready": False, "documents": [], "last_indexed": None}

async def initialize_agent():
    """异步初始化RAG Agent"""
    global agent_instance
    try:
        logger.info("🔧 正在初始化RAG Agent...")
        agent_instance = await RAGAgent.create(working_dir=WORKING_DIR)
        logger.info("✅ RAG Agent初始化完成")
        return "✅ 系统已就绪"
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return f"❌ 初始化失败: {str(e)}"

# ===== 文档索引功能 =====
async def index_documents_async(file_paths: List[str], progress=gr.Progress()):
    """异步索引文档 - 支持PDF和文本文件智能处理"""
    global index_status
    
    if not agent_instance:
        return "❌ Agent未初始化,请先启动系统", {}
    
    progress(0, desc="准备索引文档...")
    
    try:
        # 验证文件
        valid_files = [f for f in file_paths if os.path.exists(f)]
        if not valid_files:
            return "❌ 未找到有效文件", {}
        
        # 分析文件类型
        pdf_files = [f for f in valid_files if f.lower().endswith('.pdf')]
        text_files = [f for f in valid_files if f.lower().endswith(('.md', '.txt'))]
        
        progress(0.1, desc=f"检测到 {len(pdf_files)} 个PDF文件, {len(text_files)} 个文本文件")
        
        # 智能文档处理
        progress(0.3, desc=f"正在智能处理 {len(valid_files)} 个文档...")
        
        # 调用智能索引
        result = await agent_instance.index_documents(valid_files)
        
        progress(0.8, desc="索引完成,更新状态...")
        
        # 更新状态
        index_status["ready"] = True
        index_status["documents"] = [os.path.basename(f) for f in valid_files]
        index_status["last_indexed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        progress(1.0, desc="完成!")
        
        # 构建详细的指标信息
        processing_summary = result.get('processing_summary', '')
        
        metrics = {
            "索引文档数": len(valid_files),
            "PDF文件数": len(pdf_files),
            "文本文件数": len(text_files),
            "Track ID": result.get("track_id", "N/A"),
            "索引时间": index_status["last_indexed"],
            "状态": result.get("status_message", "成功")
        }
        
        # 如果有处理摘要，添加到返回信息中
        status_msg = f"✅ 成功索引 {len(valid_files)} 个文档"
        if processing_summary:
            status_msg += f"\n📊 处理摘要: {processing_summary}"
        
        return status_msg, metrics
        
    except Exception as e:
        logger.error(f"索引失败: {e}")
        return f"❌ 索引失败: {str(e)}", {}

# ===== 查询功能 =====
async def query_knowledge_async(
    question: str,
    query_mode: str,
    show_context: bool,
    chat_history: List
):
    """异步查询知识库"""
    if not agent_instance:
        return chat_history, {}, ""
    
    if not question.strip():
        return chat_history, {}, ""
    
    try:
        logger.info(f"🔍 查询: {question} (mode={query_mode})")
        
        # 执行查询
        result = await agent_instance.query(
            question=question,
            mode=query_mode
        )
        
        # 解析结果
        answer = result.get("answer", "无答案")
        context_data = result.get("context", {})
        raw_context = context_data.get("raw_context", "")
        
        # 构建回答消息
        response_msg = f"**🤖 回答** ({query_mode} 模式)\n\n{answer}"
        
        # 更新聊天历史
        chat_history.append({
            "role": "user",
            "content": question
        })
        chat_history.append({
            "role": "assistant",
            "content": response_msg
        })
        
        # 提取检索指标
        metrics = extract_metrics_from_context(raw_context, query_mode)
        
        # 格式化上下文用于显示
        formatted_context = ""
        if show_context:
            formatted_context = format_context_display(raw_context)
        
        return chat_history, metrics, formatted_context
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        error_msg = f"❌ 查询出错: {str(e)}"
        chat_history.append({
            "role": "assistant",
            "content": error_msg
        })
        return chat_history, {}, ""

# ===== 辅助函数 =====
def extract_metrics_from_context(raw_context: str, mode: str) -> Dict:
    """从上下文中提取检索指标"""
    metrics = {
        "查询模式": mode,
        "上下文长度": len(raw_context) if raw_context else 0,
    }
    
    if raw_context:
        if "Knowledge Graph Data (Entity)" in raw_context:
            entity_count = raw_context.count('{"entity":')
            metrics["图谱实体数"] = entity_count
        
        if "Document Chunks" in raw_context:
            chunk_count = raw_context.count('{"reference_id":')
            metrics["文档片段数"] = chunk_count
        
        if "Knowledge Graph Data (Relationship)" in raw_context:
            rel_count = raw_context.count('{"entity1":')
            metrics["关系三元组数"] = rel_count
    
    return metrics

def format_context_display(raw_context: str) -> str:
    """格式化上下文用于显示"""
    if not raw_context:
        return "无上下文数据"
    
    display = "### 📋 检索到的上下文\n\n"
    preview = raw_context[:1000]
    display += f"```\n{preview}\n```\n\n"
    
    if len(raw_context) > 1000:
        display += f"*... 还有 {len(raw_context) - 1000} 个字符未显示*"
    
    return display

def get_available_documents():
    """获取可用文档列表"""
    if not os.path.exists(DOC_LIBRARY):
        return []
    
    files = []
    for ext in ['*.md', '*.txt', '*.pdf']:
        files.extend(Path(DOC_LIBRARY).glob(ext))
    
    return [str(f) for f in files]

def clear_chat():
    """清空聊天"""
    return [], {}, ""

# ===== Gradio界面构建 =====
with gr.Blocks(
    title="🦙 保险文档RAG检索系统",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css
) as demo:
    
    gr.HTML("""
    <div class="header-banner">
        <h1>🦙 保险文档智能检索系统</h1>
        <p>基于 LightRAG + LangGraph 的混合检索引擎 | 支持向量检索 + 知识图谱推理</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Accordion("📁 文档库管理", open=True):
                gr.Markdown("### 索引新文档")
                
                file_input = gr.File(
                    label="上传保险条款文档 (支持PDF/MD/TXT)",
                    file_count="multiple",
                    file_types=[".md", ".txt", ".pdf"]
                )
                gr.Markdown("📋 支持PDF文件自动解析、Markdown和文本文件直接索引")
                
                with gr.Row():
                    index_btn = gr.Button("📄 开始索引", variant="primary", scale=2)
                    refresh_btn = gr.Button("🔍 查看已索引", scale=1)
                
                index_output = gr.Textbox(label="索引状态", lines=2, interactive=False)
                index_metrics = gr.JSON(label="索引统计", visible=True)
            
            with gr.Accordion("⚙️ 检索配置", open=True):
                query_mode = gr.Radio(
                    choices=[
                        ("混合检索 (推荐)", "hybrid"),
                        ("向量检索", "naive"),
                        ("局部图谱", "local"),
                        ("全局图谱", "global")
                    ],
                    value="hybrid",
                    label="检索模式"
                )
                gr.Markdown("💡 混合模式结合向量相似度和图谱推理")
                
                show_context = gr.Checkbox(
                    label="显示原始上下文",
                    value=False
                )
                gr.Markdown("📄 展示检索到的完整上下文数据")
                
                gr.Markdown("""
                **📊 检索模式说明:**
                - **混合检索**: 融合向量召回和图谱推理,准确率最高
                - **向量检索**: 纯语义相似度匹配,速度快
                - **局部图谱**: 基于实体关系的邻域搜索
                - **全局图谱**: 全图推理,适合复杂关联查询
                """)
        
        # 右侧：查询交互区
        with gr.Column(scale=7):
            gr.Markdown("### 💬 智能问答")
            
            chatbot = gr.Chatbot(
                label="对话历史",
                height=450,
                type="messages",
                avatar_images=(
                    "https://api.dicebear.com/7.x/initials/svg?seed=User",
                    "https://api.dicebear.com/7.x/bottts/svg?seed=AI"
                )
            )
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如: 什么情况下保险公司会豁免保险费?",
                    lines=2,
                    scale=8
                )
                query_btn = gr.Button("🔍 查询", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话")
                export_btn = gr.Button("💾 导出结果")
            
            # 检索指标展示
            with gr.Accordion("📊 检索质量指标", open=False):
                retrieval_metrics = gr.JSON(label="实时指标")
            
            context_display = gr.Markdown(label="原始上下文", visible=True)
    
    # 示例问题
    gr.Examples(
        examples=[
            ["什么情况下保险公司会豁免保险费?", "hybrid", False],
            ["犹豫期是多长时间?解除合同有什么后果?", "hybrid", True],
            ["全残的定义包括哪些情况?", "local", False],
            ["保险责任和责任免除有什么区别?", "global", False],
            ["投保人年龄错误会如何处理?", "naive", False],
        ],
        inputs=[query_input, query_mode, show_context],
        label="💡 示例问题 (点击快速测试)"
    )
    
    # 底部信息栏
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 8px;">
        <p style="color: #64748b; font-size: 0.9em;">
            ⚡ 技术栈: LightRAG + LangGraph + Ollama Embedding (qwen3-embedding:0.6b) + MinerU PDF解析<br>
            📚 支持文档: 寿险条款、产品说明书、理赔指南等保险文档 (PDF自动解析,MD/TXT直接索引)<br>
            🔒 数据存储: 本地向量数据库 + Neo4j知识图谱
        </p>
    </div>
    """)
    
    # ===== 事件绑定 =====
    
    # 索引事件
    index_btn.click(
        fn=index_documents_async,
        inputs=[file_input],
        outputs=[index_output, index_metrics]
    )
    
    # 查询事件
    query_btn.click(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    
    query_input.submit(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, retrieval_metrics, context_display]
    )
    
    # 导出对话
    def export_conversation(history):
        if not history:
            return "⚠️ 无对话记录"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_export_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return f"✅ 已导出至: {filename}"
    
    export_btn.click(
        fn=export_conversation,
        inputs=[chatbot],
        outputs=[query_input]
    )

# ===== 启动逻辑 =====
async def startup():
    """启动时初始化Agent"""
    print("=" * 60)
    print("🚀 正在启动保险文档RAG检索系统...")
    print("=" * 60)
    
    init_result = await initialize_agent()
    print(f"初始化结果: {init_result}")
    
    if agent_instance:
        print("\n✅ Agent初始化成功")
        print(f"📂 工作目录: {WORKING_DIR}")
        print(f"📚 文档库: {DOC_LIBRARY}")
        print("=" * 60)
    else:
        print("❌ Agent初始化失败")

if __name__ == "__main__":
    # 在 Gradio 启动前初始化 Agent
    asyncio.run(startup())
    
    # 启动 Gradio (使用 queue 启用异步支持)
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )