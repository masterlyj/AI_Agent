import gradio as gr
import asyncio
import json
import os
import base64
import time
import uuid
import threading
import html
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

from .agent import RAGAgent
from .utils import logger

# 加载环境变量
load_dotenv()

# ===== 全局配置（从环境变量读取） =====
WORKING_DIR = os.getenv("WORKING_DIR", "data/rag_storage")
DOC_LIBRARY = os.getenv("DOC_LIBRARY", "data/inputs")

# 存储模式状态
current_storage_mode = "memory"

# Rerank 配置（从环境变量读取）
def get_rerank_config():
    """从环境变量读取 Rerank 配置"""
    enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    
    if not enabled:
        return None
    
    return {
        "enabled": True,
        "model": os.getenv("RERANK_MODEL", "maidalun1020/bce-reranker-base_v1").strip(),
        "device": os.getenv("RERANK_DEVICE", "").strip() or None,
        "top_k": int(os.getenv("RERANK_TOP_K", "20")),
        "use_fp16": os.getenv("RERANK_USE_FP16", "false").lower() == "true"
    }

RERANK_CONFIG = get_rerank_config()

# 全局会话存储
user_sessions = defaultdict(lambda: {
    "thread_id": str(uuid.uuid4()),
    "chat_history": [],
    "created_at": time.time(),
    "last_active": time.time()
})

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
    max-width: 1800px !important;
    margin: 0 auto;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

/* 聊天框优化 */
.chatbot {
    height: 900px !important;
    max-height: 900px !important;
    min-height: 900px !important;
}

/* 聊天消息容器 */
.message-wrap {
    padding: 12px 16px !important;
}

/* 用户消息气泡 */
.message.user {
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
    color: white !important;
    padding: 14px 20px !important;
    border-radius: 18px 18px 4px 18px !important;
    margin: 10px 0 !important;
    max-width: 80% !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    font-weight: 400 !important;
    box-shadow: 0 3px 10px rgba(59, 130, 246, 0.25) !important;
    letter-spacing: 0.2px !important;
}

/* AI助手消息气泡 */
.message.bot {
    background: #ffffff !important;
    color: #1e293b !important;
    padding: 16px 20px !important;
    border-radius: 18px 18px 18px 4px !important;
    margin: 10px 0 !important;
    max-width: 85% !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    border-left: 4px solid var(--primary-color) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
}

/* 聊天框内的文本优化 */
.message p {
    margin: 8px 0 !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

/* 代码块样式 */
.message pre {
    background: #f8fafc !important;
    padding: 12px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    overflow-x: auto !important;
    margin: 12px 0 !important;
}

/* 思考过程特殊样式 */
.message.bot:has([class*="thinking"]) {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    border-left: 4px solid #3b82f6 !important;
}

/* Markdown标题 */
.message h1, .message h2, .message h3, .message h4 {
    margin: 16px 0 8px 0 !important;
    font-weight: 600 !important;
}

.message h1 { font-size: 20px !important; }
.message h2 { font-size: 18px !important; }
.message h3 { font-size: 16px !important; }
.message h4 { font-size: 15px !important; }

/* 列表样式 */
.message ul, .message ol {
    margin: 8px 0 !important;
    padding-left: 24px !important;
}

.message li {
    margin: 6px 0 !important;
    line-height: 1.6 !important;
}

/* 强调文本 */
.message strong {
    font-weight: 600 !important;
    color: #1e40af !important;
}

/* 分隔线 */
.message hr {
    margin: 16px 0 !important;
    border: none !important;
    border-top: 2px solid #e2e8f0 !important;
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

/* 输入框优化 */
.input-group textarea {
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
}

/* 按钮优化 */
button {
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
}

/* Tab标签优化 */
.tabs {
    font-size: 14px !important;
}

.tab-nav button {
    padding: 12px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
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

/* 深度思考区域样式 */
.thinking-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}

.thinking-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: #1e40af;
    cursor: pointer;
    font-size: 15px;
}

.thinking-icon {
    font-size: 18px;
}

.thinking-title {
    flex: 1;
}

/* 简化的深度思考内容样式 - 直接显示最新50行 */
.thinking-content-simple {
    margin-top: 12px;
    padding: 12px;
    background: white;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.6;
    color: #334155;
    max-height: none;
    word-wrap: break-word;
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

/* 可折叠侧边栏样式 - 完全隐藏 */
#sidebar-container {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

#sidebar-container.collapsed {
    width: 0px !important;
    min-width: 0px !important;
    max-width: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    opacity: 0;
}

#sidebar-container.collapsed * {
    display: none !important;
}

/* 主内容区域 */
#main-content {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* 侧边栏展开时的样式 */
#sidebar-container:not(.collapsed) {
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
}

/* 折叠按钮样式 */
.toggle-sidebar-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important;
}

.toggle-sidebar-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
}

/* 深度思考可折叠样式 - 使用HTML原生details/summary */
details.thinking-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
}

details.thinking-container summary.thinking-header {
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
    padding: 8px;
    border-radius: 8px;
    transition: background 0.2s;
    list-style: none; /* 隐藏默认的三角形 */
}

details.thinking-container summary.thinking-header::-webkit-details-marker {
    display: none; /* 隐藏WebKit浏览器的默认标记 */
}

details.thinking-container summary.thinking-header:hover {
    background: rgba(59, 130, 246, 0.1);
}

.thinking-icon {
    font-size: 20px;
    margin-right: 8px;
    transition: transform 0.3s;
}

.thinking-title {
    font-weight: 600;
    color: #1e40af;
    font-size: 15px;
}

.thinking-content {
    margin-top: 12px;
    padding: 12px;
    background: white;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.7;
    color: #1e293b;
    max-height: 400px;
    overflow-y: auto;
}

/* details展开时旋转图标 */
details.thinking-container[open] .thinking-icon {
    transform: rotate(0deg);
}

/* details关闭时旋转图标 */
details.thinking-container:not([open]) .thinking-icon {
    transform: rotate(-90deg);
}

/* 浮动配置按钮 */
.floating-config-btn {
    position: fixed;
    left: 20px;
    top: 120px;
    z-index: 1000;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 24px !important;
}

.floating-config-btn:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5) !important;
}

/* Markdown内容样式 */
.markdown-content {
    font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
    line-height: 1.7;
    color: #1e293b;
}

.markdown-content h1, .markdown-content h2, .markdown-content h3 {
    margin-top: 24px;
    margin-bottom: 16px;
    font-weight: 600;
    line-height: 1.25;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
}

.markdown-content h1 { font-size: 2em; color: #1e40af; }
.markdown-content h2 { font-size: 1.5em; color: #1e40af; }
.markdown-content h3 { font-size: 1.25em; color: #3b82f6; }

.markdown-content p {
    margin-bottom: 16px;
}

.markdown-content ul, .markdown-content ol {
    margin-bottom: 16px;
    padding-left: 2em;
}

.markdown-content li {
    margin-bottom: 8px;
}

.markdown-content code {
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 0.9em;
    color: #e11d48;
}

.markdown-content pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin-bottom: 16px;
}

.markdown-content pre code {
    background: none;
    padding: 0;
    color: inherit;
}

.markdown-content table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 16px;
}

.markdown-content table th,
.markdown-content table td {
    border: 1px solid #e2e8f0;
    padding: 12px;
    text-align: left;
}

.markdown-content table th {
    background: #f8fafc;
    font-weight: 600;
    color: #1e40af;
}

.markdown-content table tr:nth-child(even) {
    background: #f8fafc;
}

.markdown-content blockquote {
    border-left: 4px solid #3b82f6;
    padding-left: 16px;
    margin: 16px 0;
    color: #64748b;
    font-style: italic;
}

.markdown-content a {
    color: #3b82f6;
    text-decoration: none;
}

.markdown-content a:hover {
    text-decoration: underline;
}

.markdown-content hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 24px 0;
}

.markdown-content img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    margin: 16px 0;
}
"""

# ===== 初始化Agent =====
agent_instance = None
index_status = {"ready": False, "documents": [], "last_indexed": None}

async def reinitialize_agent(storage_mode):
    """重新初始化RAG Agent，使用新的存储模式"""
    global current_storage_mode, agent_instance, index_status
    
    try:
        # 清理当前实例
        agent_instance = None
        index_status = {"ready": False, "documents": [], "last_indexed": None}
        
        # 重新初始化
        result = await initialize_agent(storage_mode)
        current_storage_mode = storage_mode
        
        mode_desc = "数据库存储" if storage_mode == "database" else "内存管理"
        return f"✅ 已切换到{mode_desc}模式，系统重新初始化完成"
    except Exception as e:
        logger.error(f"❌ 重新初始化Agent失败: {e}")
        return f"❌ 重新初始化失败: {str(e)}"

async def initialize_agent(storage_mode: str = "database"):
    """异步初始化RAG Agent
    
    Args:
        storage_mode: 存储模式，可选"memory"（内存管理）或"database"（数据库存储）
    """
    global agent_instance
    try:
        logger.info("🔧 正在初始化RAG Agent...")
        agent_instance = await RAGAgent.create(
            working_dir=WORKING_DIR,
            rerank_config=RERANK_CONFIG,
            storage_mode=storage_mode
        )
        if hasattr(agent_instance, 'reranker') and agent_instance.reranker:
            logger.info(f"✅ Reranker 已加载: {RERANK_CONFIG['model']}")
        else:
            logger.warning("⚠️ Reranker 未能加载，将跳过精排步骤")
        logger.info(f"✅ RAG Agent初始化完成，存储模式: {storage_mode}")
        return f"✅ 系统已就绪，使用{storage_mode}存储模式"
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return f"❌ 初始化失败: {str(e)}"

async def index_documents_async(file_paths: List[str], progress=gr.Progress()):
    """异步索引文档 - 支持PDF和文本文件智能处理"""
    global index_status
    if not agent_instance:
        return "❌ Agent未初始化,请先启动系统", {}
    progress(0, desc="准备索引文档...")
    try:
        valid_files = [f for f in file_paths if os.path.exists(f)]
        if not valid_files:
            return "❌ 未找到有效文件", {}
        pdf_files = [f for f in valid_files if f.lower().endswith('.pdf')]
        text_files = [f for f in valid_files if f.lower().endswith(('.md', '.txt'))]
        progress(0.1, desc=f"检测到 {len(pdf_files)} 个PDF文件, {len(text_files)} 个文本文件")
        progress(0.3, desc=f"正在智能处理 {len(valid_files)} 个文档...")
        result = await agent_instance.index_documents(valid_files)
        progress(0.8, desc="索引完成,更新状态...")
        index_status["ready"] = True
        index_status["documents"] = [os.path.basename(f) for f in valid_files]
        index_status["last_indexed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress(1.0, desc="完成!")
        processing_summary = result.get('processing_summary', '')
        metrics = {
            "索引文档数": len(valid_files),
            "PDF文件数": len(pdf_files),
            "文本文件数": len(text_files),
            "Track ID": result.get("track_id", "N/A"),
            "索引时间": index_status["last_indexed"],
            "状态": result.get("status_message", "成功")
        }
        status_msg = f"✅ 成功索引 {len(valid_files)} 个文档"
        if processing_summary:
            status_msg += f"\n📊 处理摘要: {processing_summary}"
        return status_msg, metrics
    except Exception as e:
        logger.error(f"索引失败: {e}")
        return f"❌ 索引失败: {str(e)}", {}

# ===== 加载HTML模板 =====
# 全局模板缓存
_html_templates = None

# ===== 加载HTML模板（已前后端分离：改为从 frontend/ 读取静态文件） =====
_html_templates = None

def reset_html_templates_cache():
    """重置HTML模板缓存，强制重新加载文件"""
    global _html_templates
    _html_templates = None

def load_html_templates():
    """加载HTML模板配置（来自 frontend/ 目录的静态文件，返回结构与原先保持一致）"""
    global _html_templates
    if _html_templates is not None:
        return _html_templates

    from pathlib import Path
    base_dir = Path(__file__).resolve().parent / "frontend"
    html_dir = base_dir / "html"
    js_dir = base_dir / "js"

    def read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            raise FileNotFoundError(f"Missing template file: {p}. Error: {e}")

    _html_templates = {
        "knowledge_graph": {
            "template": read(html_dir / "knowledge_graph.html"),
            "script_template": read(js_dir / "knowledge_graph.js")
        },
        "document_card": {
            "template": read(html_dir / "document_card.html")
        },
        "document_container": {
            "template": read(html_dir / "document_container.html")
        },
        "empty_state": {
            "no_documents": read(html_dir / "empty_state_no_documents.html"),
            "no_context": read(html_dir / "empty_state_no_context.html"),
            "cleared": read(html_dir / "empty_state_cleared.html"),
            "loading": read(html_dir / "empty_state_loading.html")
        },
        "context_display": {
            "raw_context_template": read(html_dir / "context_display_raw_context_template.html")
        }
    }
    return _html_templates

# ===== 生成知识图谱网络可视化HTML =====
def create_knowledge_graph_html(entities, relationships, iframe_height=800):
    """
    ✅ 可直接在 Gradio 中使用的知识图谱可视化组件。
    - 根据实体类型自动分配颜色（泛化支持）
    - 节点点击显示详细信息
    - WARN 信息单独展示，不干扰图谱主视图
    """
    templates = load_html_templates()
    kg_template = templates['knowledge_graph']
    
    # 准备数据
    data_json = json.dumps({'entities': entities, 'relationships': relationships}, ensure_ascii=False)
    
    # 生成脚本内容
    script_content = kg_template['script_template'].replace('{{data_json}}', data_json)
    
    # 生成完整HTML
    page_html = kg_template['template'].replace('{{iframe_height}}', str(iframe_height))
    page_html = page_html.replace('{{script_content}}', script_content)
    
    # Base64编码
    b64 = base64.b64encode(page_html.encode("utf-8")).decode("ascii")
    iframe_html = f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:{iframe_height}px;border:none;display:block;" frameborder="0"></iframe>'
    return iframe_html


def create_documents_html(documents: List[Dict]) -> str:
    """创建文档详情可视化HTML，支持Markdown渲染"""
    import html as html_module
    try:
        import markdown
        markdown_available = True
    except ImportError:
        markdown_available = False
    
    templates = load_html_templates()
    
    if not documents:
        return templates['empty_state']['no_documents']
    
    # 生成文档卡片
    docs_html = []
    card_template = templates['document_card']['template']
    
    for idx, doc in enumerate(documents, 1):
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        # HTML 转义文件路径等元数据
        file_path = html_module.escape(metadata.get('file_path', '未知来源'))
        chunk_id = html_module.escape(str(metadata.get('chunk_id', '未知')))
        reference_id = html_module.escape(str(metadata.get('reference_id', 'N/A')))
        
        # 尝试将内容渲染为Markdown
        if markdown_available and content:
            try:
                # 使用markdown渲染，支持表格、代码块等
                content_html = markdown.markdown(
                    content, 
                    extensions=['tables', 'fenced_code', 'nl2br']
                )
                # 添加样式包装
                content_rendered = f'<div class="markdown-content">{content_html}</div>'
            except:
                # 如果渲染失败，回退到纯文本
                content_rendered = html_module.escape(content).replace('\n', '<br>')
        else:
            # 没有markdown库，使用纯文本
            content_rendered = html_module.escape(content).replace('\n', '<br>')
        
        rerank_score = metadata.get('rerank_score', 0)
        score_percent = f"{(rerank_score * 100):.2f}%" if isinstance(rerank_score, float) else "0.00%"
        
        # 替换模板占位符
        card_html = (card_template
                    .replace('{{idx}}', str(idx))
                    .replace('{{file_path}}', file_path)
                    .replace('{{chunk_id}}', chunk_id)
                    .replace('{{score_percent}}', score_percent)
                    .replace('{{reference_id}}', reference_id)
                    .replace('{{content}}', content_rendered))
        
        docs_html.append(card_html)
    
    # 生成容器HTML
    container_template = templates['document_container']['template']
    html = (container_template
           .replace('{{doc_count}}', str(len(documents)))
           .replace('{{docs_html}}', ''.join(docs_html)))
    
    return html

# ===== 查询函数,添加可视化输出 =====
async def query_knowledge_async(
    question: str,
    query_mode: str,
    show_context: bool,
    enable_rerank: bool,
    rerank_top_k: int,
    chat_history: List,
    request: gr.Request
):
    """异步查询知识库（支持流式输出，包含思考推理过程）"""
    if not agent_instance:
        yield chat_history, {}, "", "", "", "", ""
        return
    if not question.strip():
        yield chat_history, {}, "", "", "", "", ""
        return

    # 获取用户唯一标识
    session_id = request.session_hash
    user_session = user_sessions[session_id]
    thread_id = user_session["thread_id"]
    session_chat_history = user_session["chat_history"]

    logger.info(f"📌 用户会话: session_id={session_id[:8]}..., thread_id={thread_id[:8]}...")
    logger.info(f"📜 当前会话历史: {len(session_chat_history) // 2} 轮对话(共 {len(session_chat_history)} 条消息)")

    # 保存当前代理设置
    current_http_proxy = os.environ.get("HTTP_PROXY", "")
    current_https_proxy = os.environ.get("HTTPS_PROXY", "")
    current_all_proxy = os.environ.get("ALL_PROXY", "")

    # 恢复代理设置用于模型调用
    saved_http_proxy = os.environ.get("SAVED_HTTP_PROXY", "")
    saved_https_proxy = os.environ.get("SAVED_HTTPS_PROXY", "")
    saved_all_proxy = os.environ.get("SAVED_ALL_PROXY", "")

    if saved_http_proxy:
        os.environ["HTTP_PROXY"] = saved_http_proxy
    if saved_https_proxy:
        os.environ["HTTPS_PROXY"] = saved_https_proxy
    if saved_all_proxy:
        os.environ["ALL_PROXY"] = saved_all_proxy

    try:
        logger.info(f"🔍 查询: {question} (mode={query_mode}, rerank={'启用' if enable_rerank else '禁用'}, top_k={rerank_top_k})")

        # 添加加载状态
        templates = load_html_templates()
        loading_html = templates['empty_state']['loading']

        # 先添加用户消息到历史
        display_chat_history = session_chat_history + [{"role": "user", "content": question}]
        # 立即清空查询框（返回空字符串）
        yield display_chat_history, {}, "", loading_html, "", "", ""

        # 使用流式查询
        context_data = {}
        entities = []
        relationships = []
        documents = []
        raw_context = ""
        accumulated_reasoning = ""
        accumulated_answer = ""
        
        async for chunk in agent_instance.query_stream(
            question=question,
            mode=query_mode,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
            chat_history=session_chat_history,
            thread_id=thread_id,
        ):
            chunk_type = chunk.get("type")
            
            if chunk_type == "status":
                # 显示状态更新
                status_msg = chunk.get("content", "")
                logger.info(f"📊 状态: {status_msg}")
                # 可以选择在界面上显示状态（这里暂时跳过）
                
            elif chunk_type == "context":
                # 接收上下文数据
                context_data = chunk.get("context", {})
                raw_context = context_data.get("raw_context", "")
                entities = context_data.get("entities", [])
                relationships = context_data.get("relationships", [])
                documents = context_data.get("documents", [])
                
                # 生成可视化内容
                kg_html = create_knowledge_graph_html(entities, relationships)
                docs_html = create_documents_html(documents)
                
                # 更新指标
                rerank_status = "✅ 已精排" if enable_rerank and hasattr(agent_instance, 'reranker') and agent_instance.reranker else "⚠️ 未精排"
                metrics = {
                    "查询模式": query_mode,
                    "实体数量": len(entities),
                    "关系数量": len(relationships),
                    "文档片段": len(documents),
                    "精排状态": rerank_status,
                    "精排Top-K": rerank_top_k if enable_rerank else "N/A",
                    "上下文长度": len(raw_context),
                    "会话ID": session_id[:8] + "...",
                    "线程ID": thread_id[:8] + "...",
                    "对话轮数": (len(session_chat_history) + 2) // 2
                }
                
                formatted_context = ""
                if show_context:
                    formatted_context = format_context_display(raw_context)
                
                # 显示"正在思考..."占位符
                thinking_placeholder = "🧠 **正在思考...**\n\n_分析问题中..._"
                current_chat = display_chat_history + [{"role": "assistant", "content": thinking_placeholder}]
                logger.info("📝 显示思考占位符")
                yield current_chat, metrics, formatted_context, kg_html, docs_html, "", ""
                
            elif chunk_type == "reasoning_chunk":
                # 流式接收思考推理过程，显示在聊天框中
                content = chunk.get("content", "")
                is_done = chunk.get("done", False)
                
                # 累积内容
                if content:
                    accumulated_reasoning += content
                
                 # 在聊天框中显示思考过程（使用HTML原生details/summary标签，流式输出时默认展开）
                if accumulated_reasoning:
                    # 将内容分行并反向显示，让最新的内容在顶部
                    lines = accumulated_reasoning.split('\n')
                    # 只显示最后50行，避免内容过长
                    display_lines = lines[-50:] if len(lines) > 50 else lines
                    display_content = '<br>'.join(display_lines)
                    
                    thinking_html = f"""<details class="thinking-container" open>
<summary class="thinking-header">
<span class="thinking-icon">🔽</span>
<span class="thinking-title">深度思考 (实时 - 最新50行)</span>
<span style="margin-left: auto; color: #64748b; font-size: 13px;">{len(accumulated_reasoning)} 字符 / {len(lines)} 行</span>
</summary>
<div class="thinking-content-simple">
{display_content}
<div style="margin-top: 8px; padding: 8px; background: #f0f9ff; border-radius: 4px; font-size: 12px; color: #64748b; text-align: center;">
⬇️ 最新内容 ⬇️
</div>
</div>
</details>"""
                    thinking_message = thinking_html
                else:
                    thinking_message = """<div class="thinking-container">
<div class="thinking-header">
<span class="thinking-icon">🧠</span>
<span class="thinking-title">正在思考...</span>
</div>
</div>"""
                
                current_chat = display_chat_history + [{"role": "assistant", "content": thinking_message}]
                
                # 思考推理阶段，只更新聊天框，其他组件保持不变（使用gr.update()）
                yield current_chat, gr.update(), gr.update(), gr.update(), gr.update(), "", ""
                
            elif chunk_type == "answer_chunk":
                # 流式接收答案片段
                content = chunk.get("content", "")
                accumulated_answer += content
                
                # # 第一次收到答案时，记录日志
                # if len(accumulated_answer) == len(content):
                #     logger.info(f"🎯 开始生成答案，深度思考已完成 (思考长度: {len(accumulated_reasoning)} 字符)")
                
                # 答案生成时，保留折叠的深度思考，然后显示答案
                if accumulated_reasoning:
                    # 生成折叠的深度思考HTML（默认关闭，使用details标签）
                    thinking_collapsed_html = f"""<details class="thinking-container">
<summary class="thinking-header">
<span class="thinking-icon">🔽</span>
<span class="thinking-title">深度思考</span>
<span style="margin-left: auto; color: #64748b; font-size: 13px;">{len(accumulated_reasoning)} 字符</span>
</summary>
<div class="thinking-content">
{accumulated_reasoning.replace(chr(10), '<br>')}
</div>
</details>

---

{accumulated_answer}"""
                    current_chat = display_chat_history + [{"role": "assistant", "content": thinking_collapsed_html}]
                else:
                    current_chat = display_chat_history + [{"role": "assistant", "content": accumulated_answer}]
                
                # 答案流式输出时，不重新生成HTML组件，使用gr.update()保持不变
                # 只更新聊天框，其他组件保持不变
                yield current_chat, gr.update(), gr.update(), gr.update(), gr.update(), "", ""
                
            elif chunk_type == "complete":
                # 查询完成
                final_answer = chunk.get("answer", accumulated_answer)
                updated_chat_history = chunk.get("chat_history", [])
                context_data = chunk.get("context", context_data)
                
                # 【关键修复】：保存纯文本历史到session（用于后端LLM推理）
                # 不要将深度思考HTML保存到chat_history中，避免传递给LLM造成干扰
                user_session["chat_history"] = updated_chat_history
                user_session["last_active"] = time.time()
                
                # 【显示用】：生成包含深度思考的HTML版本，仅用于前端显示
                display_updated_history = []
                for msg in updated_chat_history:
                    display_updated_history.append(msg.copy())
                
                # 如果有深度思考，在最后一条assistant消息前添加深度思考HTML（仅用于显示）
                if accumulated_reasoning and display_updated_history:
                    for i in range(len(display_updated_history) - 1, -1, -1):
                        if display_updated_history[i].get("role") == "assistant":
                            # 生成折叠的深度思考HTML（默认关闭，使用details标签）
                            thinking_collapsed_html = f"""<details class="thinking-container">
<summary class="thinking-header">
<span class="thinking-icon">🔽</span>
<span class="thinking-title">深度思考</span>
<span style="margin-left: auto; color: #64748b; font-size: 13px;">{len(accumulated_reasoning)} 字符</span>
</summary>
<div class="thinking-content">
{accumulated_reasoning.replace(chr(10), '<br>')}
</div>
</details>

---

{display_updated_history[i].get('content', '')}"""
                            display_updated_history[i]["content"] = thinking_collapsed_html
                            break
                
                # 最终更新
                rerank_status = "✅ 已精排" if enable_rerank and hasattr(agent_instance, 'reranker') and agent_instance.reranker else "⚠️ 未精排"
                metrics = {
                    "查询模式": query_mode,
                    "实体数量": len(entities),
                    "关系数量": len(relationships),
                    "文档片段": len(documents),
                    "精排状态": rerank_status,
                    "精排Top-K": rerank_top_k if enable_rerank else "N/A",
                    "上下文长度": len(raw_context),
                    "会话ID": session_id[:8] + "...",
                    "线程ID": thread_id[:8] + "...",
                    "对话轮数": len(updated_chat_history) // 2
                }
                
                formatted_context = ""
                if show_context:
                    formatted_context = format_context_display(raw_context)
                
                kg_html = create_knowledge_graph_html(entities, relationships)
                docs_html = create_documents_html(documents)
                
                # 完成时显示深度思考的完整内容（使用包含深度思考HTML的显示版本）
                yield display_updated_history, metrics, formatted_context, kg_html, docs_html, "", ""
                
            elif chunk_type == "error":
                # 处理错误
                error_msg = chunk.get("content", "未知错误")
                logger.error(f"流式查询出错: {error_msg}")
                
                error_chat_history = session_chat_history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                ]
                user_session["chat_history"] = error_chat_history
                
                yield error_chat_history, {}, "", "", "", "", ""
                return

    except Exception as e:
        logger.error(f"查询失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        error_msg = f"❌ 查询出错: {str(e)}"

        # 错误时也要正确更新历史
        error_chat_history = session_chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_msg}
        ]
        user_session["chat_history"] = error_chat_history

        yield error_chat_history, {}, "", "", "", "", ""
        return
    finally:
        # 恢复查询前的代理设置
        os.environ["HTTP_PROXY"] = current_http_proxy
        os.environ["HTTPS_PROXY"] = current_https_proxy
        os.environ["ALL_PROXY"] = current_all_proxy

def extract_metrics_from_context(raw_context: str, mode: str) -> Dict:
    """从上下文中提取检索指标，支持多种数据格式"""
    metrics = {
        "查询模式": mode,
        "上下文长度": len(raw_context) if raw_context else 0,
    }
    
    # 调试输出
    print(f"DEBUG - 开始提取指标，模式: {mode}, 上下文长度: {metrics['上下文长度']}")
    
    if not raw_context:
        print("DEBUG - 上下文为空，返回基础指标")
        return metrics
    
    # 尝试解析JSON格式的上下文
    try:
        # 检查是否为JSON格式
        if raw_context.strip().startswith('{') or raw_context.strip().startswith('['):
            import json
            parsed_data = json.loads(raw_context)
            
            # 如果是字典格式，直接提取实体和关系
            if isinstance(parsed_data, dict):
                entities = parsed_data.get("entities", [])
                relationships = parsed_data.get("relationships", [])
                documents = parsed_data.get("documents", [])
                
                metrics["图谱实体数"] = len(entities) if isinstance(entities, list) else 0
                metrics["关系三元组数"] = len(relationships) if isinstance(relationships, list) else 0
                metrics["文档片段数"] = len(documents) if isinstance(documents, list) else 0
                
                print(f"DEBUG - JSON格式解析成功: 实体{metrics['图谱实体数']}个, 关系{metrics['关系三元组数']}个, 文档{metrics['文档片段数']}个")
                return metrics
            
            # 如果是列表格式，假设是文档列表
            elif isinstance(parsed_data, list):
                metrics["文档片段数"] = len(parsed_data)
                print(f"DEBUG - 检测到文档列表格式: {metrics['文档片段数']}个文档")
                return metrics
    except json.JSONDecodeError:
        print("DEBUG - JSON解析失败，使用文本计数方式")
    except Exception as e:
        print(f"DEBUG - 解析过程中出现错误: {e}")
    
    # 传统文本计数方式（向后兼容）
    entity_count = raw_context.count('{"entity":') + raw_context.count('"entity_name"')
    chunk_count = raw_context.count('{"reference_id":') + raw_context.count('"content"')
    rel_count = raw_context.count('{"entity1":') + raw_context.count('"src_id"') + raw_context.count('"source"')
    
    metrics["图谱实体数"] = entity_count
    metrics["文档片段数"] = chunk_count
    metrics["关系三元组数"] = rel_count
    
    print(f"DEBUG - 文本计数完成: 实体{entity_count}个, 文档{chunk_count}个, 关系{rel_count}个")
    
    return metrics

def format_context_display(raw_context: str) -> str:
    """格式化原始上下文显示，支持Markdown渲染"""
    try:
        import markdown
        markdown_available = True
    except ImportError:
        markdown_available = False
    
    templates = load_html_templates()
    
    if not raw_context:
        return templates['empty_state']['no_context']
    
    raw_template = templates['context_display']['raw_context_template']
    
    # 尝试将内容渲染为Markdown
    if markdown_available and raw_context:
        try:
            # 使用markdown渲染，支持表格、代码块等
            content_html = markdown.markdown(
                raw_context, 
                extensions=['tables', 'fenced_code', 'nl2br', 'codehilite']
            )
            # 添加样式包装
            content_rendered = f'<div class="markdown-content">{content_html}</div>'
        except:
            # 如果渲染失败，回退到纯文本
            content_rendered = html.escape(raw_context).replace('\n', '<br>')
    else:
        # 没有markdown库，使用纯文本
        content_rendered = html.escape(raw_context).replace('\n', '<br>')
    
    return (raw_template
           .replace('{{char_count}}', str(len(raw_context)))
           .replace('{{content}}', content_rendered))

def _create_context_html(entities: List[Dict], relationships: List[Dict]) -> str:
    """创建实体和关系的HTML显示"""
    import random
    import colorsys
    
    def generate_color_for_type(entity_type: str) -> str:
        """为实体类型生成一致的颜色，支持无限种类型"""
        # 使用实体类型的哈希值生成0-1之间的浮点数
        hash_value = hash(entity_type) % 10000 / 10000.0
        
        # 使用HSV颜色空间生成饱和度高、亮度适中的颜色
        # 色相根据哈希值变化，饱和度和亮度固定在合适范围
        hue = hash_value
        saturation = 0.7 + (hash_value * 0.3)  # 0.7-1.0之间，确保颜色鲜艳
        value = 0.6 + (hash_value * 0.3)  # 0.6-0.9之间，确保不会太亮或太暗
        
        # 转换为RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 转换为十六进制颜色代码
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    html = """
    <div style="font-family: 'Microsoft YaHei', sans-serif; padding: 16px; background: #f8fafc; border-radius: 8px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1e293b; font-size: 24px;">📊 检索上下文</h2>
            <div style="margin-left: auto; display: flex; gap: 16px;">
                <div style="background: #3b82f6; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;">
                    实体: {entity_count}
                </div>
                <div style="background: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;">
                    关系: {relationship_count}
                </div>
            </div>
        </div>
    """.format(entity_count=len(entities), relationship_count=len(relationships))
    
    if entities:
        html += """
        <div style="margin-bottom: 24px;">
            <h3 style="color: #1e40af; margin-bottom: 12px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">🔍</span> 实体信息
            </h3>
            <div style="display: grid; gap: 12px;">
        """
        
        for i, entity in enumerate(entities[:10]):
            name = entity.get('entity_name', entity.get('name', '未知实体'))
            entity_type = entity.get('entity_type', entity.get('type', '未知类型'))
            description = entity.get('description', entity.get('desc', '无描述'))
            
            # 为每个实体类型生成一致的颜色
            type_color = generate_color_for_type(entity_type)
            
            html += f"""
                <div style="background: white; padding: 12px; border-radius: 8px; border-left: 4px solid {type_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="font-weight: bold; color: #1e293b; font-size: 16px;">{name}</div>
                        <div style="background: {type_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">{entity_type}</div>
                    </div>
                    <div style="color: #64748b; font-size: 14px; line-height: 1.5;">{description}</div>
                </div>
            """
        
        if len(entities) > 10:
            html += f"""
                <div style="text-align: center; color: #64748b; font-size: 14px; padding: 8px;">
                    ... 还有 {len(entities) - 10} 个实体未显示
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    if relationships:
        html += """
        <div>
            <h3 style="color: #059669; margin-bottom: 12px; display: flex; align-items: center;">
                <span style="margin-right: 8px;">🔗</span> 关系信息
            </h3>
            <div style="display: grid; gap: 12px;">
        """
        
        for i, rel in enumerate(relationships[:10]):
            src = rel.get('src_id', rel.get('source', rel.get('from', '未知源')))
            tgt = rel.get('tgt_id', rel.get('target', rel.get('to', '未知目标')))
            weight = rel.get('weight', rel.get('score', 0))
            description = rel.get('description', rel.get('desc', rel.get('relation', '无描述')))
            
            weight_color = '#10b981' if weight > 0.8 else '#f59e0b' if weight > 0.5 else '#ef4444'
            
            html += f"""
                <div style="background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="font-weight: bold; color: #1e293b; font-size: 16px;">{src}</div>
                            <div style="color: #10b981; font-size: 18px;">→</div>
                            <div style="font-weight: bold; color: #1e293b; font-size: 16px;">{tgt}</div>
                        </div>
                        <div style="background: {weight_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{weight:.2f}</div>
                    </div>
                    <div style="color: #64748b; font-size: 14px; line-height: 1.5;">{description}</div>
                </div>
            """
        
        if len(relationships) > 10:
            html += f"""
                <div style="text-align: center; color: #64748b; font-size: 14px; padding: 8px;">
                    ... 还有 {len(relationships) - 10} 个关系未显示
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    if not entities and not relationships:
        html += """
        <div style="text-align: center; padding: 40px; color: #64748b; background: white; border-radius: 8px; border: 2px dashed #cbd5e1;">
            <div style="font-size: 48px; margin-bottom: 16px;">📭</div>
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">暂无上下文数据</div>
            <div>请先执行查询以获取实体和关系信息</div>
        </div>
        """
    
    html += "</div>"
    return html

def _create_documents_html(documents: List[Dict]) -> str:
    """创建文档列表的HTML显示"""
    html = """
    <div style="font-family: 'Microsoft YaHei', sans-serif; padding: 16px; background: #f8fafc; border-radius: 8px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1e293b; font-size: 24px;">📄 检索文档</h2>
            <div style="margin-left: auto; background: #3b82f6; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;">
                {doc_count} 个文档
            </div>
        </div>
        <div style="display: grid; gap: 12px;">
    """.format(doc_count=len(documents))
    
    for i, doc in enumerate(documents[:10]):
        content = doc.get('content', doc.get('text', '无内容'))
        metadata = doc.get('metadata', {})
        file_path = metadata.get('file_path', '未知来源')
        
        html += f"""
        <div style="background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-weight: bold; color: #1e293b; font-size: 16px;">📁 {file_path}</div>
                <div style="background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">#{i+1}</div>
            </div>
            <div style="color: #64748b; font-size: 14px; line-height: 1.5; max-height: 100px; overflow-y: auto;">{content[:200]}{'...' if len(content) > 200 else ''}</div>
        </div>
        """
    
    if len(documents) > 10:
        html += f"""
        <div style="text-align: center; color: #64748b; font-size: 14px; padding: 8px;">
            ... 还有 {len(documents) - 10} 个文档未显示
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    return html

def get_available_documents():
    """获取可用文档列表"""
    if not os.path.exists(DOC_LIBRARY):
        return []
    files = []
    for ext in ['*.md', '*.txt', '*.pdf']:
        files.extend(Path(DOC_LIBRARY).glob(ext))
    return [str(f) for f in files]

def clear_chat(request: gr.Request):
    """清空当前用户的对话历史"""
    session_id = request.session_hash
    if session_id in user_sessions:
        # 重新生成 thread_id 和清空历史
        user_sessions[session_id] = {
            "thread_id": str(uuid.uuid4()),
            "chat_history": [],
            "created_at": time.time(),
            "last_active": time.time()
        }
        logger.info(f"🗑️ 已清空用户 {session_id[:8]}... 的会话")
    # 返回6个输出：chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization, reasoning_display
    return [], {}, "", "<p style='text-align:center; color:#999;'>已清空</p>", "<p style='text-align:center; color:#999;'>已清空</p>", ""

def cleanup_inactive_sessions():
    """定期清理 30 分钟未活动的会话"""
    while True:
        time.sleep(600)  # 每 10 分钟检查一次
        current_time = time.time()
        inactive_timeout = 1800  # 30 分钟
        
        inactive_sessions = [
            sid for sid, data in list(user_sessions.items())
            if current_time - data.get("last_active", current_time) > inactive_timeout
        ]
        
        for sid in inactive_sessions:
            del user_sessions[sid]
            logger.info(f"🧹 清理不活跃会话: {sid[:8]}... (超过 30 分钟未活动)")

# 启动清理线程
cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
cleanup_thread.start()

# ===== Gradio界面构建 =====
with gr.Blocks(
    title="🦙 保险文档RAG检索系统",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=custom_css
) as demo:
    # 简洁的标题栏
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 28px; font-weight: 600; color: #1e293b;">🦙 保险智能问答</h1>
        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 14px;">基于知识图谱的智能检索系统</p>
    </div>
    """)
    
    with gr.Row():
        # 浮动配置按钮
        toggle_sidebar_btn = gr.Button("⚙️", elem_classes=["floating-config-btn"], visible=True)
        
        with gr.Column(scale=1, visible=False) as sidebar_column:
            # 侧边栏标题
            gr.Markdown("## ⚙️ 系统配置")
            close_sidebar_btn = gr.Button("✕ 关闭配置", elem_id="close-sidebar-btn", variant="secondary")
            
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
                
            with gr.Accordion("💾 存储配置", open=True):
                storage_mode = gr.Radio(
                    choices=[
                        ("数据库存储 (推荐)", "database"),
                        ("内存管理 (轻量)", "memory")
                    ],
                    value="database",
                    label="存储模式",
                    info="数据库存储适合生产环境，内存管理适合快速测试"
                )
                reinit_btn = gr.Button("🔄 应用存储模式", variant="secondary")
                storage_status = gr.Textbox(label="存储状态", lines=1, interactive=False)
                
            with gr.Accordion("⚙️ 检索配置", open=True):
                query_mode = gr.Radio(
                    choices=[
                        ("综合检索(推荐)", "mix"),
                        ("传统向量检索", "naive"),
                        ("实体聚焦检索", "local"),
                        ("关系聚焦检索", "global"),
                        ("混合检索", "hybrid")
                    ],
                    value="mix",
                    label="检索模式"
                )
                gr.Markdown("💡 Mix模式融合知识图谱和向量检索，提供最全面的检索结果")
                enable_rerank_checkbox = gr.Checkbox(
                    label="✅ 启用精排 (Rerank)",
                    value=True,
                    info="对向量检索结果进行二次排序, 提高精度 (仅对混合/向量模式有效)"
                )
                rerank_top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    label="📊 精排Top-K文档数",
                    info="精排后返回的文档数量，数值越大返回结果越多但可能引入噪声"
                )
                show_context = gr.Checkbox(
                    label="显示原始上下文",
                    value=False
                )
                gr.Markdown("📄 展示检索到的完整上下文数据")
                gr.Markdown("""
                **📊 检索模式说明:**
                - **综合检索**: 融合知识图谱和向量检索，提供最全面的检索结果
                - **传统向量检索**: 纯语义相似度匹配，速度快
                - **实体聚焦检索**: 基于实体关系的邻域搜索
                - **关系聚焦检索**: 全图推理，适合复杂关联查询
                - **混合检索**: 结合local和global两种策略
                """)
        with gr.Column(scale=7, elem_id="main-content"):
            chatbot = gr.Chatbot(
                height=900,
                type="messages",
                show_label=False,
                avatar_images=(
                    "https://api.dicebear.com/7.x/initials/svg?seed=User",
                    "https://api.dicebear.com/7.x/bottts/svg?seed=AI"
                )
            )
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="输入问题，例如: 什么情况下保险公司会豁免保险费?",
                    lines=2,
                    scale=8,
                    show_label=False
                )
                query_btn = gr.Button("🔍", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话")
                export_btn = gr.Button("💾 导出结果")
            with gr.Accordion("📊 检索质量指标", open=False):
                retrieval_metrics = gr.JSON(label="实时指标")
            # ===== 可视化标签页 =====
            with gr.Tabs():
                with gr.Tab("🕸️ 知识图谱"):
                    kg_visualization = gr.HTML(
                        label="知识图谱可视化",
                        value="<p style='text-align:center; color:#999;'>执行查询后将显示知识图谱</p>"
                    )
                with gr.Tab("📄 文档详情"):
                    docs_visualization = gr.HTML(
                        label="精排文档详情",
                        value="<p style='text-align:center; color:#999;'>执行查询后将显示文档详情</p>"
                    )
                with gr.Tab("📝 原始上下文"):
                    context_display = gr.HTML(
                        label="原始上下文",
                        value="执行查询后将显示原始上下文"
                    )
            
            # 隐藏的占位组件（用于兼容事件绑定）
            reasoning_display = gr.HTML(visible=False)
    gr.Examples(
        examples=[
            ["什么情况下保险公司会豁免保险费?", "hybrid", False, True, 20],
            ["犹豫期是多长时间?解除合同有什么后果?", "hybrid", True, True, 15],
            ["全残的定义包括哪些情况?", "local", False, True, 10],
            ["保险责任和责任免除有什么区别?", "global", False, True, 20],
            ["投保人年龄错误会如何处理?", "naive", False, True, 20],
        ],
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox, rerank_top_k_slider],
        label="💡 示例问题 (点击快速测试)"
    )
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 8px;">
        <p style="color: #64748b; font-size: 0.9em;">
            ⚡ 技术栈: LightRAG + LangGraph + Ollama Embedding (qwen3-embedding:0.6b) + MinerU PDF解析<br>
            📚 支持文档: 寿险条款、产品说明书、理赔指南等保险文档 (PDF自动解析,MD/TXT直接索引)<br>
            💾 数据存储: 支持数据库存储和内存管理两种模式，可在配置面板中切换
        </p>
    </div>
    """)
    # ===== 事件绑定 =====
    index_btn.click(
        fn=index_documents_async,
        inputs=[file_input],
        outputs=[index_output, index_metrics]
    )
    
    # 重新初始化Agent事件
    reinit_btn.click(
        fn=reinitialize_agent,
        inputs=[storage_mode],
        outputs=[storage_status]
    )
    
    query_btn.click(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox, rerank_top_k_slider, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization, reasoning_display, query_input]
    )
    
    query_input.submit(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox, rerank_top_k_slider, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization, reasoning_display, query_input]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization, reasoning_display]
    )
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
    
    # 侧边栏折叠/展开事件 - 优化响应速度
    toggle_sidebar_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[sidebar_column, toggle_sidebar_btn],
        queue=False
    )
    
    close_sidebar_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[sidebar_column, toggle_sidebar_btn],
        queue=False
    )

# ===== 启动逻辑 =====
async def startup():
    print("=" * 60)
    print("🚀 正在启动保险文档RAG检索系统...")
    print("=" * 60)
    
    # 重置HTML模板缓存，确保使用最新的文件
    reset_html_templates_cache()
    print("✅ 已重置HTML模板缓存")
    
    # 使用默认存储模式初始化
    init_result = await initialize_agent(current_storage_mode)
    print(f"初始化结果: {init_result}")
    if agent_instance:
        print("\n✅ Agent初始化成功")
        print(f"📂 工作目录: {WORKING_DIR}")
        print(f"📚 文档库: {DOC_LIBRARY}")
        print(f"💾 存储模式: {current_storage_mode}")
        print("=" * 60)
    else:
        print("❌ Agent初始化失败")

if __name__ == "__main__":
    # 在 Gradio 启动前初始化 Agent
    asyncio.run(startup())

    # 保存当前代理设置
    current_http_proxy = os.environ.get("HTTP_PROXY", "")
    current_https_proxy = os.environ.get("HTTPS_PROXY", "")
    current_all_proxy = os.environ.get("ALL_PROXY", "")
    
    # 将当前代理设置保存到SAVED_*_PROXY环境变量，供模型调用时使用
    os.environ["SAVED_HTTP_PROXY"] = current_http_proxy
    os.environ["SAVED_HTTPS_PROXY"] = current_https_proxy
    os.environ["SAVED_ALL_PROXY"] = current_all_proxy
    
    print(f"当前代理设置: HTTP_PROXY={current_http_proxy}, HTTPS_PROXY={current_https_proxy}, ALL_PROXY={current_all_proxy}")
    print("禁用代理以启动 Gradio 共享链接...")
    # 禁用代理以确保Gradio能够创建共享链接
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["ALL_PROXY"] = ""
    
    # 启动 Gradio (使用 queue 启用异步支持)
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )
    
    # 恢复原始代理设置（在Gradio关闭后）
    os.environ["HTTP_PROXY"] = current_http_proxy
    os.environ["HTTPS_PROXY"] = current_https_proxy
    os.environ["ALL_PROXY"] = current_all_proxy
    print("已恢复原始代理设置")