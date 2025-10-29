# HTML模板配置说明

## 📁 文件说明

本目录包含以下文件：
- `html_templates.json` - HTML模板配置文件
- `insurance_rag_gradio.py` - 主程序文件
- `HTML_TEMPLATES_README.md` - 本说明文档

## 🎯 设计目标

将HTML代码从Python代码中分离出来，实现：
1. **代码分离** - HTML模板与业务逻辑分离
2. **易于维护** - 修改样式不需要修改Python代码
3. **可重用性** - 模板可以被多个函数复用
4. **代码美观** - Python代码更简洁易读

## 📝 模板结构

### 1. 知识图谱模板 (`knowledge_graph`)

```json
{
  "knowledge_graph": {
    "template": "完整的HTML页面模板",
    "script_template": "JavaScript脚本模板"
  }
}
```

**占位符说明：**
- `{{iframe_height}}` - iframe高度（像素）
- `{{script_content}}` - JavaScript代码内容
- `{{data_json}}` - 实体和关系的JSON数据

### 2. 文档卡片模板 (`document_card`)

```json
{
  "document_card": {
    "template": "单个文档卡片的HTML"
  }
}
```

**占位符说明：**
- `{{idx}}` - 文档序号
- `{{file_path}}` - 文件路径
- `{{chunk_id}}` - 文档块ID
- `{{score_percent}}` - 相关度百分比（格式：XX.XX%）
- `{{reference_id}}` - 引用ID
- `{{content}}` - 文档内容（已HTML转义）

### 3. 文档容器模板 (`document_container`)

```json
{
  "document_container": {
    "template": "文档列表容器HTML"
  }
}
```

**占位符说明：**
- `{{doc_count}}` - 文档总数
- `{{docs_html}}` - 所有文档卡片的HTML（拼接后的字符串）

### 4. 空状态模板 (`empty_state`)

```json
{
  "empty_state": {
    "no_documents": "无文档提示",
    "no_context": "无上下文提示",
    "cleared": "已清空提示",
    "loading": "加载中动画"
  }
}
```

### 5. 上下文显示模板 (`context_display`)

```json
{
  "context_display": {
    "raw_context_template": "原始上下文显示模板"
  }
}
```

**占位符说明：**
- `{{char_count}}` - 字符数
- `{{content}}` - 上下文内容

## 🔧 如何修改模板

### 修改样式

1. 打开 `html_templates.json`
2. 找到对应的模板
3. 修改CSS样式或HTML结构
4. 保存文件（程序会自动重新加载）

**示例：修改文档卡片边框颜色**

```json
{
  "document_card": {
    "template": "<div style='... border:2px solid #10b981; ...'>"
  }
}
```

将 `#10b981`（绿色）改为 `#3b82f6`（蓝色）即可。

### 添加新模板

1. 在 `html_templates.json` 中添加新的键值对
2. 在Python代码中使用 `load_html_templates()` 读取
3. 使用 `.replace()` 方法替换占位符

**示例：**

```python
templates = load_html_templates()
my_template = templates['my_new_template']['template']
html = my_template.replace('{{placeholder}}', 'value')
```

## 💡 最佳实践

### 1. 占位符命名规范
- 使用双花括号：`{{variable_name}}`
- 使用小写字母和下划线
- 命名要有描述性

### 2. HTML转义
- 用户输入的内容必须使用 `html.escape()` 转义
- 模板中的静态内容不需要转义

```python
import html as html_module
safe_content = html_module.escape(user_input)
```

### 3. 模板缓存
- 程序会缓存模板到全局变量 `_html_templates`
- 首次加载后不会重复读取文件
- 如需重新加载，重启程序即可

### 4. 模板维护
- 保持模板的独立性，避免相互依赖
- 使用注释说明复杂的HTML结构
- 定期检查模板是否符合最新的需求

## 🎨 样式定制指南

### 颜色系统

当前使用的主要颜色：
- 主色调：`#1e40af` (蓝色)
- 成功色：`#10b981` (绿色)
- 警告色：`#f59e0b` (橙色)
- 危险色：`#ef4444` (红色)
- 渐变色板：`#667eea`, `#f093fb`, `#4facfe`, 等

### 响应式设计

模板使用 flexbox 布局，支持响应式：
- 使用 `flex: 1` 自适应宽度
- 使用 `max-width` 限制最大宽度
- 使用 `overflow-y: auto` 添加滚动条

### 字体大小

- 标题：18-24px
- 正文：14px
- 小字：12px
- 标签：11-13px

## 🐛 故障排除

### 问题：修改模板后没有生效
**解决方案：** 重启程序以重新加载模板

### 问题：页面显示乱码
**解决方案：** 检查 `html_templates.json` 文件编码是否为 UTF-8

### 问题：占位符没有被替换
**解决方案：** 
1. 检查占位符拼写是否正确
2. 确保使用 `.replace('{{name}}', value)` 方法
3. 检查是否有多个同名占位符

### 问题：JSON格式错误
**解决方案：**
1. 使用 JSON 验证工具检查语法
2. 注意转义特殊字符（`\n` 在JSON中需要写作 `\\n`）
3. 确保所有字符串使用双引号

## 📚 相关函数

### Python代码中使用模板的函数

1. `load_html_templates()` - 加载模板配置
2. `create_knowledge_graph_html()` - 生成知识图谱HTML
3. `create_documents_html()` - 生成文档详情HTML
4. `format_context_display()` - 格式化上下文显示
5. `clear_chat()` - 清空聊天界面

## 🔄 版本历史

- **v1.0** (2025-01-28) - 初始版本，实现HTML模板分离

## 📞 支持

如有问题，请查看：
1. 本文档的故障排除部分
2. 代码中的注释说明
3. JSON文件中的模板结构

---

**注意：** 修改模板时请务必备份原文件，以防出现问题时可以恢复。

