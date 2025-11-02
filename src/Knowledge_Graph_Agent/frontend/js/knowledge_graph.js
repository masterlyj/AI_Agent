(function() {
    const dataFromPy = {{data_json}};
    // 日志函数
    const log = (msg, type = 'info') => {
        console.log(msg);
        if (type === 'warn') return;
        const el = document.getElementById('log');
        const line = document.createElement('div');
        line.className = type === 'success' ? 'log-success' : 'log-line';
        line.textContent = msg;
        el.appendChild(line);
        el.scrollTop = el.scrollHeight;
    };

    // 配色及类型颜色映射
    const palette = [
        "#667eea", "#f093fb", "#4facfe", "#00f2fe", "#43e97b",
        "#fa709a", "#fee140", "#30cfd0", "#a8edea", "#fbc2eb"
    ];
    const typeColorMap = {};
    const getColor = (type) => {
        if (!typeColorMap[type]) {
            const keys = Object.keys(typeColorMap);
            typeColorMap[type] = palette[keys.length % palette.length];
        }
        return typeColorMap[type];
    };

    // 节点处理
    const nodes = [];
    const name2id = {};
    (dataFromPy.entities || []).forEach((ent, i) => {
        const id = i + 1;
        const name = (ent.entity_name || ent.name || ("Entity_" + i)).trim();
        const type = (ent.entity_type || "Unknown").trim();
        const desc = (ent.description || "无描述").toString();
        const color = getColor(type);
        name2id[name] = id;
        nodes.push({
            id,
            label: name,
            title: type,
            color: { background: color, border: "#555" },
            data: {
                name,
                type,
                description: desc,
                source_id: ent.source_id || "",
                file_path: ent.file_path || "",
                created_at: ent.created_at || ""
            }
        });
    });

    // 关系处理
    const edges = [];
    (dataFromPy.relationships || []).forEach((rel) => {
        const src = (rel.src_id || rel.source || "").trim();
        const tgt = (rel.tgt_id || rel.target || "").trim();
        const from = name2id[src];
        const to = name2id[tgt];
        if (!from || !to) {
            log("⚠️ 关系跳过：实体不存在 - " + src + " → " + tgt, 'warn');
            return;
        }
        edges.push({
            from,
            to,
            label: (rel.keywords || "").toString().slice(0, 30),
            arrows: "to",
            color: { color: "#999" },
            font: {
                align: "horizontal",
                size: 11,
                color: "#2d3748",
                background: "rgba(255, 255, 255, 0.9)",
                strokeWidth: 0,
                strokeColor: "transparent"
            }
        });
    });

    // 初始化网络
    const container = document.getElementById("network");
    const data = {
        nodes: new vis.DataSet(nodes),
        edges: new vis.DataSet(edges)
    };
    const options = {
        nodes: {
            shape: "dot",
            size: 22,
            font: {
                size: 14,
                color: "#2d3748",
                face: "Microsoft YaHei, Arial",
                bold: { color: "#1a202c" }
            },
            borderWidth: 2,
            borderWidthSelected: 3,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.15)',
                size: 8,
                x: 2,
                y: 2
            }
        },
        edges: {
            smooth: {
                enabled: true,
                type: "continuous",
                roundness: 0.5
            },
            color: {
                color: "#cbd5e0",
                highlight: "#667eea",
                hover: "#667eea"
            },
            width: 2,
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.8
                }
            },
            font: {
                size: 11,
                color: "#2d3748",
                strokeWidth: 0,
                align: "horizontal",
                background: "rgba(255, 255, 255, 0.9)",
                vadjust: -8
            },
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.08)',
                size: 3,
                x: 1,
                y: 1
            }
        },
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 200,
                updateInterval: 25
            },
            barnesHut: {
                gravitationalConstant: -10000,
                centralGravity: 0.3,
                springLength: 150,
                springConstant: 0.04,
                damping: 0.09,
                avoidOverlap: 0.5
            }
        },
        interaction: {
            hover: true,
            hoverConnectedEdges: true,
            selectConnectedEdges: true,
            tooltipDelay: 200,
            navigationButtons: true,
            keyboard: true
        }
    };
    const network = new vis.Network(container, data, options);

    // 图谱稳定后自动fit视图
    network.once('stabilizationIterationsDone', function() {
        network.fit();
    });

    // 加载完成日志
    log("✅ 知识图谱加载成功：" + nodes.length + " 个实体，" + edges.length + " 条关系", 'success');

    // 侧栏显示实体详情
    const sidebar = document.getElementById("entityDetails");
    network.on("click", function(params) {
        if (params.nodes.length === 0) return;
        const nodeId = params.nodes[0];
        const node = data.nodes.get(nodeId);
        if (node && node.data) {
            const info = node.data;
            sidebar.innerHTML =
                "<div class='info-item'>" +
                    "<b>名称</b><div>" + (info.name || '未知') + "</div>" +
                "</div>" +
                "<div class='info-item'>" +
                    "<b>类型</b><div>" + (info.type || '未知') + "</div>" +
                "</div>" +
                "<div class='info-item'>" +
                    "<b>描述</b><div>" + (info.description || '无描述') + "</div>" +
                "</div>" +
                "<div class='info-item'>" +
                    "<b>来源 chunk</b><div>" + (info.source_id || '未知') + "</div>" +
                "</div>" +
                "<div class='info-item'>" +
                    "<b>文档路径</b><div style='word-break: break-all;'>" + (info.file_path || '未知') + "</div>" +
                "</div>" +
                "<div class='info-item'>" +
                    "<b>创建时间</b><div>" + (info.created_at || '未知') + "</div>" +
                "</div>";
        }
    });
})()