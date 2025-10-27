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

#新增reranker配置
# RERANK_CONFIG = {
#     "enabled": True,
#     "model": "maidalun1020/bce-reranker-base_v1",  # 支持 HuggingFace 模型名或本地路径
#     "device": None,  # 仅 HuggingFace 使用
#     "top_k": 3
# }

# 本地模型加载示例（可选配置）
RERANK_CONFIG = {
    "enabled": True,
    "model": "D:/Codes/modelscope/bce-reranker-base_v1",  # 本地模型路径
    "device": None,  # 指定GPU设备
    "top_k": 20
}

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
        agent_instance = await RAGAgent.create(
            working_dir=WORKING_DIR,
            rerank_config=RERANK_CONFIG
        )
        if hasattr(agent_instance, 'reranker') and agent_instance.reranker:
            logger.info(f"✅ Reranker 已加载: {RERANK_CONFIG['model']}")
        else:
            logger.warning("⚠️ Reranker 未能加载，将跳过精排步骤")
        logger.info("✅ RAG Agent初始化完成")
        return "✅ 系统已就绪"
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

# ===== 生成知识图谱网络可视化HTML =====
def create_knowledge_graph_html(entities: List[Dict], relationships: List[Dict]) -> str:
    """创建基于 vis.js 的知识图谱网络可视化"""
    if not entities and not relationships:
        return "<div style='text-align:center; color:#666; padding:40px; background:#f8fafc; border-radius:8px; border:2px dashed #cbd5e1;'><h3>📋 暂无知识图谱数据</h3><p>请先执行查询以获取知识图谱数据</p></div>"
    
    # 调试输出数据格式
    print(f"DEBUG - 实体数据示例: {entities[0] if entities else 'None'}")
    print(f"DEBUG - 关系数据示例: {relationships[0] if relationships else 'None'}")
    
    # 将数据转换为 JSON，确保中文字符正确显示
    data_json = json.dumps({
        "entities": entities,
        "relationships": relationships
    }, ensure_ascii=False)
    
    # 使用多个CDN源以提高加载成功率
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 尝试多个CDN源加载vis.js -->
    <script>
        // 尝试加载vis.js，如果失败则尝试备用源
        function loadVisJS() {{
            console.log('开始加载vis.js...');
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js';
            script.onload = function() {{
                console.log('vis.js loaded successfully from unpkg');
                initNetwork();
            }};
            script.onerror = function() {{
                console.log('Failed to load vis.js from unpkg, trying jsdelivr...');
                const fallbackScript = document.createElement('script');
                fallbackScript.src = 'https://cdn.jsdelivr.net/npm/vis-network@9.1.2/dist/vis-network.min.js';
                fallbackScript.onload = function() {{
                    console.log('vis.js loaded successfully from jsdelivr');
                    initNetwork();
                }};
                fallbackScript.onerror = function() {{
                    console.error('Failed to load vis.js from all sources');
                    document.getElementById('network').innerHTML = '<div style="padding:20px;text-align:center;color:red;">无法加载可视化库，请检查网络连接</div>';
                }};
                document.head.appendChild(fallbackScript);
            }};
            document.head.appendChild(script);
        }}
        
        // 初始化网络图
        function initNetwork() {{
            try {{
                console.log('开始初始化网络图...');
                const data = {data_json};
                
                // 检查数据格式并适配
                console.log('Entities count:', data.entities.length);
                console.log('Relationships count:', data.relationships.length);
                
                if (data.entities.length === 0 && data.relationships.length === 0) {{
                    document.getElementById('network').innerHTML = '<div style="padding:20px;text-align:center;color:#666;">暂无知识图谱数据</div>';
                    return;
                }}
                
                // 准备节点数据 - 适配不同的数据格式
                const entityNameToId = {{}};
                const nodesArray = [];
                
                for (let i = 0; i < data.entities.length; i++) {{
                    const entity = data.entities[i];
                    // 适配不同的实体字段名
                    const name = entity.entity_name || entity.name || entity.id || `Entity_${{i}}`;
                    const type = entity.entity_type || entity.type || '未知类型';
                    let description = entity.description || entity.desc || '无描述';
                    
                    // 处理描述中的特殊字符
                    description = description.replace(/<SEP>/g, ' ').substring(0, 200);
                    
                    entityNameToId[name] = i;
                    
                    // 根据实体类型设置不同颜色
                    let nodeColor = '#3b82f6'; // 默认蓝色
                    if (type.includes('保险') || type.includes('Insurance')) {{
                        nodeColor = '#10b981'; // 绿色
                    }} else if (type.includes('疾病') || type.includes('Disease')) {{
                        nodeColor = '#ef4444'; // 红色
                    }} else if (type.includes('时间') || type.includes('Time')) {{
                        nodeColor = '#f59e0b'; // 橙色
                    }}
                    
                    nodesArray.push({{
                        id: i,
                        label: name,
                        title: `<b>${{name}}</b><br>类型: ${{type}}<br>描述: ${{description}}`,
                        type: type,
                        description: description,
                        color: {{
                            background: nodeColor,
                            border: '#1e293b',
                            highlight: {{ background: nodeColor, border: '#1e293b' }}
                        }},
                        font: {{ color: '#ffffff', size: 14, bold: true }},
                        shape: 'dot',
                        size: 20 + Math.min(description.length / 20, 15) // 根据描述长度调整节点大小
                    }});
                }}
                
                console.log('Created', nodesArray.length, 'nodes');
                
                // 准备边数据 - 适配不同的关系字段名
                const edgesArray = [];
                
                for (let i = 0; i < data.relationships.length; i++) {{
                    const rel = data.relationships[i];
                    // 适配不同的关系字段名
                    const src = rel.src_id || rel.source || rel.from;
                    const tgt = rel.tgt_id || rel.target || rel.to;
                    const weight = rel.weight || rel.score || 1.0;
                    let description = rel.description || rel.desc || rel.relation || '无描述';
                    
                    // 处理描述中的特殊字符
                    description = description.replace(/<SEP>/g, ' ').substring(0, 200);
                    
                    const fromId = entityNameToId[src];
                    const toId = entityNameToId[tgt];
                    
                    if (fromId === undefined || toId === undefined) {{
                        console.warn(`无法找到关系中的实体: ${{src}} -> ${{tgt}}`);
                        continue;
                    }}
                    
                    edgesArray.push({{
                        id: i,
                        from: fromId,
                        to: toId,
                        label: `${{typeof weight === 'number' ? weight.toFixed(2) : weight}}`,
                        title: description,
                        arrows: 'to',
                        color: {{ color: '#10b981', highlight: '#059669' }},
                        width: Math.max(1, Math.min(weight * 3, 5)), // 根据权重调整边宽度
                        font: {{ size: 11, align: 'middle' }},
                        smooth: {{ type: 'cubicBezier', roundness: 0.3 }}
                    }});
                }}
                
                console.log('Created', edgesArray.length, 'edges');
                
                // 创建数据集
                const nodes = new vis.DataSet(nodesArray);
                const edges = new vis.DataSet(edgesArray);
                
                // 配置选项
                const options = {{
                    nodes: {{ 
                        borderWidth: 2, 
                        shadow: true,
                        font: {{
                            color: '#ffffff',
                            size: 14,
                            face: 'Microsoft YaHei'
                        }}
                    }},
                    edges: {{ 
                        shadow: true,
                        font: {{
                            color: '#1e293b',
                            size: 11,
                            face: 'Microsoft YaHei'
                        }}
                    }},
                    physics: {{
                        enabled: true,
                        stabilization: {{ iterations: 200 }},
                        barnesHut: {{
                            gravitationalConstant: -8000,
                            springConstant: 0.04,
                            springLength: 150
                        }}
                    }},
                    interaction: {{ 
                        hover: true, 
                        tooltipDelay: 100,
                        navigationButtons: true,
                        keyboard: true
                    }}
                }};
                
                // 创建网络
                console.log('Creating vis.Network...');
                const container = document.getElementById('network');
                const network = new vis.Network(container, {{ nodes, edges }}, options);
                console.log('Network created successfully');
                
                // 点击节点显示详情
                network.on('click', function(params) {{
                    const infoPanel = document.getElementById('node-info');
                    if (params.nodes.length > 0) {{
                        const node = nodes.get(params.nodes[0]);
                        document.getElementById('info-title').textContent = `🏷️ ${{node.label}}`;
                        document.getElementById('info-content').innerHTML = `
                            <div><b>类型:</b> ${{node.type}}</div>
                            <div style="margin-top:8px;"><b>描述:</b> ${{node.description}}</div>
                        `;
                        infoPanel.classList.add('show');
                    }} else {{
                        infoPanel.classList.remove('show');
                    }}
                }});
                
                // 稳定后停止物理模拟
                network.once('stabilizationIterationsDone', function() {{
                    network.setOptions({{ physics: false }});
                    console.log('Network stabilized');
                }});
                
                // 添加缩放控制
                network.fit();
                
            }} catch (error) {{
                console.error('Error initializing network:', error);
                document.getElementById('network').innerHTML = `<div style="padding:20px;text-align:center;color:red;">初始化知识图谱时出错: ${{error.message}}</div>`;
            }}
        }}
        
        // 页面加载完成后加载vis.js
        window.onload = loadVisJS;
    </script>
    <style>
        #network {{
            width: 100%;
            height: 600px;
            border: 2px solid #3b82f6;
            border-radius: 12px;
            background: #f8fafc;
            position: relative;
        }}
        
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            font-family: 'Microsoft YaHei', sans-serif;
        }}
        
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #1e293b;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 8px 0;
            font-size: 13px;
        }}
        
        .node-info {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: none;
            font-family: 'Microsoft YaHei', sans-serif;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .node-info.show {{
            display: block;
        }}
        
        .info-title {{
            color: #1e40af;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .info-content {{
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 600px;
            flex-direction: column;
        }}
        
        .spinner {{
            border: 4px solid #f3f4f6;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-bottom: 10px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body style="margin:0; position:relative;">
    <div id="network">
        <div class="loading">
            <div class="spinner"></div>
            <p>正在加载知识图谱...</p>
        </div>
    </div>
    
    <div class="legend">
        <div class="legend-title">📋 图例</div>
        <div class="legend-item">
            <div style="width:20px;height:20px;border-radius:50%;background:#3b82f6;"></div>
            <span>默认实体</span>
        </div>
        <div class="legend-item">
            <div style="width:20px;height:20px;border-radius:50%;background:#10b981;"></div>
            <span>保险相关</span>
        </div>
        <div class="legend-item">
            <div style="width:20px;height:20px;border-radius:50%;background:#ef4444;"></div>
            <span>疾病相关</span>
        </div>
        <div class="legend-item">
            <div style="width:20px;height:20px;border-radius:50%;background:#f59e0b;"></div>
            <span>时间相关</span>
        </div>
        <div class="legend-item">
            <div style="width:40px;height:3px;background:#10b981;"></div>
            <span>关系连线</span>
        </div>
        <div style="margin-top:10px;font-size:12px;color:#64748b;">
            💡 点击节点查看详情<br>
            🖱️ 拖拽可移动节点<br>
            🔍 滚轮可缩放
        </div>
    </div>
    
    <div id="node-info" class="node-info">
        <div id="info-title" class="info-title"></div>
        <div id="info-content" class="info-content"></div>
    </div>
</body>
</html>
    """
    return html

def create_documents_html(documents: List[Dict]) -> str:
    """创建文档详情可视化HTML"""
    if not documents:
        return "<p style='text-align:center; color:#666; padding:40px;'>暂无文档数据</p>"
    
    docs_html = []
    for idx, doc in enumerate(documents, 1):
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        file_path = metadata.get('file_path', '未知来源')
        chunk_id = metadata.get('chunk_id', '未知')
        rerank_score = metadata.get('rerank_score', 0)
        reference_id = metadata.get('reference_id', 'N/A')
        
        score_percent = (rerank_score * 100) if isinstance(rerank_score, float) else 0
        
        docs_html.append(f"""
        <div style='background:#f8fafc; border:2px solid #10b981; border-radius:10px; 
                    padding:20px; margin-bottom:15px; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                <div style='background:#10b981; color:white; width:35px; height:35px; 
                            border-radius:50%; display:flex; align-items:center; 
                            justify-content:center; font-weight:bold;'>{idx}</div>
                <div style='flex:1;'>
                    <div style='font-weight:bold; color:#065f46; font-size:15px;'>
                        📁 {file_path}
                    </div>
                    <div style='margin-top:6px; display:flex; gap:10px; flex-wrap:wrap;'>
                        <span style='background:#dbeafe; color:#1e40af; padding:4px 10px; 
                                     border-radius:6px; font-size:12px;'>🔖 {chunk_id}</span>
                        <span style='background:#fef3c7; color:#d97706; padding:4px 10px; 
                                     border-radius:6px; font-size:12px; font-weight:600;'>
                            📈 相关度: {score_percent:.2f}%
                        </span>
                        <span style='background:#f3e8ff; color:#7c3aed; padding:4px 10px; 
                                     border-radius:6px; font-size:12px;'>
                            🆔 {reference_id}
                        </span>
                    </div>
                </div>
            </div>
            <div style='background:white; padding:15px; border-radius:8px; 
                        border:1px solid #e5e7eb; margin-top:12px;'>
                <div style='color:#1f2937; line-height:1.8; white-space:pre-wrap; font-size:14px;'>
                    {content}
                </div>
            </div>
        </div>
        """)
    
    html = f"""
    <div style='padding:20px; max-height:650px; overflow-y:auto; font-family:"Microsoft YaHei", sans-serif;'>
        <h3 style='color:#047857; margin-bottom:20px; display:flex; align-items:center;'>
            <span style='font-size:24px; margin-right:10px;'>📄</span>
            精排文档详情 - {len(documents)} 个文档片段
        </h3>
        {''.join(docs_html)}
    </div>
    """
    return html

# ===== 查询函数,添加可视化输出 =====
async def query_knowledge_async(
    question: str,
    query_mode: str,
    show_context: bool,
    enable_rerank: bool,
    chat_history: List
):
    """异步查询知识库"""
    if not agent_instance:
        return chat_history, {}, "", "", ""
    if not question.strip():
        return chat_history, {}, "", "", ""
    try:
        logger.info(f"🔍 查询: {question} (mode={query_mode}, rerank={'启用' if enable_rerank else '禁用'})")
        result = await agent_instance.query(
            question=question,
            mode=query_mode,
            enable_rerank=enable_rerank
        )
        answer = result.get("answer", "无答案")
        context_data = result.get("context", {})
        raw_context = context_data.get("raw_context", "")
        entities = context_data.get("entities", [])
        relationships = context_data.get("relationships", [])
        documents = context_data.get("documents", [])
        kg_html = create_knowledge_graph_html(entities, relationships)
        docs_html = create_documents_html(documents)
        rerank_status = "✅ 已精排" if enable_rerank and hasattr(agent_instance, 'reranker') and agent_instance.reranker else "⚠️ 未精排"
        response_msg = f"**🤖 回答** ({query_mode} 模式 | {rerank_status})\n\n{answer}"
        chat_history.append({
            "role": "user",
            "content": question
        })
        chat_history.append({
            "role": "assistant",
            "content": response_msg
        })
        metrics = {
            "查询模式": query_mode,
            "实体数量": len(entities),
            "关系数量": len(relationships),
            "文档片段": len(documents),
            "精排状态": rerank_status,
            "上下文长度": len(raw_context)
        }
        formatted_context = ""
        if show_context:
            formatted_context = format_context_display(raw_context)
        return chat_history, metrics, formatted_context, kg_html, docs_html
    except Exception as e:
        logger.error(f"查询失败: {e}")
        error_msg = f"❌ 查询出错: {str(e)}"
        chat_history.append({
            "role": "assistant",
            "content": error_msg
        })
        return chat_history, {}, "", "", ""

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
    """格式化上下文用于显示"""
    if not raw_context:
        return "<div style='text-align:center; color:#999; padding:40px;'>📭 暂无上下文数据</div>"
    
    # 尝试解析JSON格式的上下文
    try:
        import json
        if raw_context.strip().startswith('{') or raw_context.strip().startswith('['):
            context_data = json.loads(raw_context)
            
            # 如果是字典格式，提取实体和关系
            if isinstance(context_data, dict):
                entities = context_data.get("entities", [])
                relationships = context_data.get("relationships", [])
                return _create_context_html(entities, relationships)
            # 如果是列表格式，假设是文档列表
            elif isinstance(context_data, list):
                return _create_documents_html(context_data)
    except (json.JSONDecodeError, Exception):
        pass
    
    # 如果不是JSON格式，使用原始显示方式
    return f"""
    <div style="font-family: 'Microsoft YaHei', sans-serif; padding: 16px; background: #f8fafc; border-radius: 8px;">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h3 style="margin: 0; color: #1e293b; font-size: 20px;">📄 原始上下文</h3>
            <div style="margin-left: auto; background: #3b82f6; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;">
                {len(raw_context)} 字符
            </div>
        </div>
        <details style="margin-top: 16px;">
            <summary style="cursor: pointer; color: #3b82f6; font-weight: bold; padding: 8px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">点击展开/折叠原始上下文</summary>
            <pre style="margin-top: 12px; padding: 16px; background: white; border-radius: 6px; border: 1px solid #e2e8f0; overflow-x: auto; font-size: 14px; line-height: 1.6;">{raw_context}</pre>
        </details>
    </div>
    """

def _create_context_html(entities: List[Dict], relationships: List[Dict]) -> str:
    """创建实体和关系的HTML显示"""
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
            
            type_color = '#3b82f6'
            if '保险' in entity_type or 'Insurance' in entity_type:
                type_color = '#10b981'
            elif '疾病' in entity_type or 'Disease' in entity_type:
                type_color = '#ef4444'
            elif '时间' in entity_type or 'Time' in entity_type:
                type_color = '#f59e0b'
            
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

def clear_chat():
    return [], {}, "", "<p style='text-align:center; color:#999;'>已清空</p>", "<p style='text-align:center; color:#999;'>已清空</p>"

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
        with gr.Column(scale=7):
            gr.Markdown("### 💬 智能问答")
            chatbot = gr.Chatbot(
                label="对话历史",
                height=400,
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
            with gr.Accordion("📊 检索质量指标", open=False):
                retrieval_metrics = gr.JSON(label="实时指标")
            # ===== 新增: 可视化标签页 =====
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
                    context_display = gr.Markdown(
                        label="原始上下文",
                        value="执行查询后将显示原始上下文"
                    )
    gr.Examples(
        examples=[
            ["什么情况下保险公司会豁免保险费?", "hybrid", False, True],
            ["犹豫期是多长时间?解除合同有什么后果?", "hybrid", True, True],
            ["全残的定义包括哪些情况?", "local", False, True],
            ["保险责任和责任免除有什么区别?", "global", False, True],
            ["投保人年龄错误会如何处理?", "naive", False, True],
        ],
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox],
        label="💡 示例问题 (点击快速测试)"
    )
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
    index_btn.click(
        fn=index_documents_async,
        inputs=[file_input],
        outputs=[index_output, index_metrics]
    )
    query_btn.click(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    query_input.submit(
        fn=query_knowledge_async,
        inputs=[query_input, query_mode, show_context, enable_rerank_checkbox, chatbot],
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, retrieval_metrics, context_display, kg_visualization, docs_visualization]
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

# ===== 启动逻辑 =====
async def startup():
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
        share=True,
        debug=True,
        show_error=True
    )