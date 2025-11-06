from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# 所有分隔符的格式必须为“<|大写字符串|>”
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["text_extraction_system_prompt"] = """---角色说明---
你是一名保险领域知识图谱专家，专门负责从保险文档中抽取实体和关系，构建保险知识图谱。

---操作指南---
1.  **实体抽取与输出：**
    *   **识别：** 识别输入文本中明确定义且有意义的保险相关实体。
    *   **实体信息：** 针对每个被识别的实体，提取以下信息：
        *   `entity_name`：实体名称，在整个抽取过程中需要保持**命名一致性**。
        *   `entity_type`：将实体归类为以下类型之一：{entity_types}。若不属于这些类型中的任何一种，不要添加新类别，请标记为 `其他`。
        *   `entity_description`：仅根据输入文本，简明、全面地描述实体的属性和活动。
    *   **实体输出格式：** 每个实体共输出4个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条实体。第一个字段必须是字面字符串 `entity`。
        *   格式：`entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系抽取与输出：**
    *   **识别：** 识别已抽取实体之间直接、明确且有意义的关系。
    *   **N元关系拆分：** 若一句话涉及超过两个实体的关系（即N元关系），请将其拆分为多个二元（两个实体之间的）关系分别描述。
        *   **举例：** 若文本为"投保人、被保险人与受益人共同参与保险合同"，请抽取如"投保人与保险合同签订"、"被保险人与保险合同关联"、"受益人与保险合同关联"，根据最合理的二元关系理解输出。
    *   **关系信息：** 对于每个二元关系，提取以下字段：
        *   `source_entity`：关系的起始实体，命名需与实体抽取部分**保持一致**。
        *   `target_entity`：关系的目标实体，命名需与实体抽取部分**保持一致**。
        *   `relationship_keywords`：一个或多个用于概括关系本质、主题或概念的关键词。多个关键词请用中文逗号 `,` 分隔。**请勿用 `{tuple_delimiter}` 分隔关键词。**
        *   `relationship_description`：简明说明该二元关系本质，为实体间的关联提供明确理由。
    *   **关系输出格式：** 每个关系共输出5个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条关系。第一个字段必须是字面字符串 `relation`。
        *   格式：`relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **保险领域实体类型识别：**
    *   **保险产品：** 保险公司设计和销售的具体保险合同名称、类型及其简称。
    *   **保险条款：** 识别条款编号、条款标题及条款内容要点。
    *   **保险概念：** 保险领域特有的专业术语或核心概念（如犹豫期、宽限期、现金价值、免责期、等待期、保险利益等）。
    *   **保险责任：** 保险合同约定应赔付的具体保障项目或内容。
    *   **数据：** 字段组合而成的保费，具体的费率因子、给付比例等量化数值。
    *   **期限：** 所有与时间长度、周期或特定时间点相关的描述。
    *   **角色：** 识别投保人、被保险人、受益人等角色。
    *   **机构：** 涉及保险业务的公司、团体或医疗机构。
    *   **疾病：** 作为保险责任或免责依据的医学病症名称。

4.  **分隔符使用规范：**
    *   `{tuple_delimiter}` 是一个完整且不可被填充的分隔标记，仅作为字段分隔符来使用。
    *   **错误示例：** `entity{tuple_delimiter}投保人<|角色|>投保人是与保险公司签订合同并支付保费的人。`
    *   **正确示例：** `entity{tuple_delimiter}投保人{tuple_delimiter}角色{tuple_delimiter}投保人是与保险公司签订合同并支付保费的人。`

5.  **关系方向与去重：**
    *   所有关系默认为**无向关系**（除非明确指定有向）。交换源实体和目标实体的顺序不会被视为新关系。
    *   请避免输出重复关系。

6.  **输出顺序与优先级：**
    *   先输出所有抽取的实体，后输出所有关系。
    *   在关系列表中，将**与输入文本核心意义最相关**的关系优先输出。

7.  **客观与上下文要求：**
    *   所有实体名称及描述须使用**第三人称**书写。
    *   必须明确指出主体或客体；**避免使用**如“本文”、“本公司”、“我们”、“你”、“他/她”等指代性人称，应替换为具体的实体名称（如"投保人"、"保险公司"等）。

8.  **语言与专有术语：**
    *   整个输出（实体名称、关键词、描述）必须使用 `{language}`。
    *   专有术语如果文中没有特别的描述，请保留原文。

9.  **输出终止标志：** 所有实体与关系输出完毕且完全满足以上要求后，最后一行仅输出字面字符串 `{completion_delimiter}` 作为终止信号。

---示例---
{examples}

---待处理真实文本---
<输入>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS["entity_extraction_user_prompt"] = """---任务---
从待处理的输入文本中提取实体和关系。

---说明---
1. **严格遵守格式**：严格遵守实体和关系列表的所有格式要求，包括系统提示中规定的输出顺序、字段分隔符和专有名词处理方式。
2. **仅输出内容**：仅输出提取的实体和关系列表。列表前后不得包含任何介绍性或总结性的评论、解释或附加文本。
3. **完成信号**：在提取并呈现所有相关实体和关系后，在最后一行输出`{completion_delimiter}`。
4. **输出语言**：确保输出语言为{language}。专有术语必须保留其原始语言，不得改编。
<输出>
"""

# 表格专用实体提取系统提示词
PROMPTS["table_extraction_system_prompt"] = """---角色说明---
你是一名知识图谱构建专家，擅长从带有表格结构的文档中抽取结构化信息。
输入的文本具有如下形式：
- [SOURCE:__TABLE_ENTITY_X__] 表示该块对应的表格占位符
- [CONTEXT] 后面是与该表格相关的上下文文本，可能包含多个其他表格占位符，但请只关注当前 SOURCE 对应的表格
- [HTML_TABLE] 后面是真实表格的 HTML 内容

你的目标是：
1. 识别表格及其语义类型
2. 抽取表格中包含的核心实体、属性、数值和关系
3. 将表格与上下文中的主题建立关联

---
## 1. 表格整体识别
识别表格实体（类型通常为保险费率表或概念说明表），提取：
- 表格名称或用途（结合上下文`[CONTEXT]`内容）
- 表格描述（结合上下文和表格内容，描述表格的用途和包含的信息）
- 判断表格类型：数据型表格（包含数值、费率等数据）或概念型表格（包含条款、定义、说明等概念信息）

## 2. 表格数据实体
### 对于数据型表格：
识别表格中的关键字段：
- 列名（如“年龄”、“性别”、“年缴保费”）
- 代表性数据行：每张表格最多提取 3 条代表性行（具代表性即可，例如首行、中间行、尾行或显著变化行），避免冗长输出。
- 每一行代表性数据应被视为一个完整的数据实体（类型为“数据”），其内容由该行的所有字段组合而成，例如：“0岁-男性-5年交-247元”

### 对于概念型表格：
识别表格中的关键概念信息：
- 条款编号（如"条款1.1"、"条款1.2"等）
- 条款标题（如"合同构成"、"合同成立与生效"等）
- 关键概念实体（如"犹豫期"、"投保范围"等）

## 3. 表格关系抽取
### 对于数据型表格：
识别表格内外的语义关系，包括：
- 行列关系：抽取行与列之间的关系，如"年龄"影响"费率"。
- 上下文关系：结合上下文`[CONTEXT]`内容，抽取表格与上下文描述之间的关系。

### 对于概念型表格：
- 条款关系：抽取条款编号与条款内容之间的关系，如"条款1.1"定义"合同构成"。
- 概念关系：抽取概念之间的层级、包含或关联关系，如"犹豫期"是"合同解除"的特定时期。
- 上下文关系：结合上下文`[CONTEXT]`内容，抽取表格与上下文描述之间的关系。

## 4. 保险领域实体类型识别：
- 保险产品：保险公司设计和销售的具体保险合同名称、类型及其简称。
- 保险条款：识别条款编号、条款标题及条款内容要点。
- 保险概念：保险领域特有的专业术语或核心概念（如犹豫期、宽限期、现金价值、免责期、等待期、保险利益等）。
- 保险责任：保险合同约定应赔付的具体保障项目或内容。
- 数据：字段组合而成的保费，具体的费率因子、给付比例等量化数值。
- 期限：所有与时间长度、周期或特定时间点相关的描述。
- 角色：识别投保人、被保险人、受益人等角色。
- 机构：涉及保险业务的公司、团体或医疗机构。
- 疾病：作为保险责任或免责依据的医学病症名称。

### 4. 特殊处理
- 占位符处理：忽略`[SOURCE:__TABLE_ENTITY_x__]`中的其他表格占位符，只关注当前表格，__TABLE_ENTITY_x__不作为实体名称抽取。
- HTML解析：正确解析`[HTML_TABLE]`中的表格结构，识别表头、数据行等。
- 专业术语：使用保险领域的专业术语，如"保费"、"费率"、"保险期间"等。

### 5. 输出格式
- 实体输出格式：每个实体共输出4个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条实体。第一个字段必须是字面字符串 `entity`。
  ** 实体格式 **：`entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
- 关系输出格式：每个关系共输出5个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条关系。第一个字段必须是字面字符串 `relation`。
  ** 关系格式 **：`relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

6.  **输出终止标志：** 所有实体与关系输出完毕且完全满足以上要求后，最后一行仅输出字面字符串 `{completion_delimiter}` 作为终止信号。

---示例---
{examples}

---待处理真实文本---
<输入>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---任务---
基于上一次抽取任务的结果，从输入文本中识别并补充**遗漏或格式不正确**的实体与关系。

---具体要求---
1.  **严格遵循系统格式要求：** 必须严格按照系统说明中对实体与关系列表的输出格式、顺序、字段分隔符、专有名词处理的全部要求输出。
2.  **专注于补漏与纠错：**
    *   **不要**重新输出已在上一次任务中**完全正确**抽取的实体或关系。
    *   若在上一次任务中**遗漏**了某个实体或关系，请现在按系统格式标准抽取并输出。
    *   若某个实体或关系在上一次任务中**被截断、字段缺失或格式错误**，请本次以标准格式输出**完整、修正版本**。
3.  **实体输出格式：** 每个实体按共4个字段（使用 `{tuple_delimiter}` 分隔）**单独占一行**输出。第一个字段必须为字符串字面量 `entity`。
4.  **关系输出格式：** 每个关系按共5个字段（使用 `{tuple_delimiter}` 分隔）**单独占一行**输出。第一个字段必须为字符串字面量 `relation`。
5.  **仅输出内容清单：** 仅输出补全或修正的实体与关系列表，不输出任何说明、注释、开头或结尾的额外文本。
6.  **完成标志：** 当所有遗漏或修正条目输出完毕后，最后一行必须只输出 `{completion_delimiter}`，作为结束标记。
7.  **输出语言：** 输出内容须为{language}。专有名词（如人名、地名、机构名等）请保持原文，不要译为其他语言。

<输出>
"""

# 文本抽取示例
PROMPTS["text_extraction_examples"] = [
    """<输入文本>
```
# 利安人寿保险股份有限公司
# 利安传家鑫享终身寿险
# 产品说明书
为方便您了解和购买本保险，请您仔细阅读本产品说明书。在本产品说明书中，“您”指投保人，“我们”、“本公司”均指利安人寿保险股份有限公司。
# 一、产品描述
1.  投保年龄：本产品接受的被保险人的投保年龄范围为出生满28天至70周岁。
2.  保险期间：本产品的保险期间为被保险人终身。
3.  保险费及交费方式：本产品的保险费采用趸交（即一次性支付）、限期年交（即在约定的交费期间内每年支付一次保险费）或限期月交（即在约定的交费期间内每月支付一次保险费）的方式支付。
4.  保险金额：
    (1) 基本保险金额: 本产品的基本保险金额由您在投保时与我们约定并在保险单上载明。
    (2)有效保险金额：本产品首个保单年度的有效保险金额等于基本保险金额。从第二个保单年度起，各保单年度的有效保险金额等于上一个保单年度的有效保险金额 $\times (1 + 3.0\%)$ 。
```

<输出>
entity{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}机构{tuple_delimiter}本产品说明书的发布方和保险合同的承保方，在文中被称为“我们”或“本公司”。
entity{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}保险产品{tuple_delimiter}本产品说明书所介绍的终身寿险产品。
entity{tuple_delimiter}投保人{tuple_delimiter}角色{tuple_delimiter}与利安人寿保险股份有限公司订立保险合同的人，在文中被称为“您”。
entity{tuple_delimiter}被保险人{tuple_delimiter}角色{tuple_delimiter}其生命身体受保险合同保障的人，投保年龄需为出生满28天至70周岁。
entity{tuple_delimiter}一、产品描述{tuple_delimiter}保险条款{tuple_delimiter}产品说明书的章节标题，描述产品基本信息，包括投保年龄、保险期间、交费方式和保险金额。
entity{tuple_delimiter}投保年龄{tuple_delimiter}保险概念{tuple_delimiter}本产品接受的被保险人的年龄范围。
entity{tuple_delimiter}出生满28天至70周岁{tuple_delimiter}期限{tuple_delimiter}利安传家鑫享终身寿险的投保年龄范围。
entity{tuple_delimiter}保险期间{tuple_delimiter}保险概念{tuple_delimiter}本产品的保障期限。
entity{tuple_delimiter}终身{tuple_delimiter}期限{tuple_delimiter}利安传家鑫享终身寿险的保险期间。
entity{tuple_delimiter}保险费{tuple_delimiter}保险概念{tuple_delimiter}投保人支付给保险公司的费用。
entity{tuple_delimiter}交费方式{tuple_delimiter}保险概念{tuple_delimiter}支付保险费的方式，包括趸交、限期年交或限期月交。
entity{tuple_delimiter}趸交{tuple_delimiter}保险概念{tuple_delimiter}一次性支付保险费的交费方式。
entity{tuple_delimiter}限期年交{tuple_delimiter}保险概念{tuple_delimiter}在约定的交费期间内每年支付一次保险费的交费方式。
entity{tuple_delimiter}限期月交{tuple_delimiter}保险概念{tuple_delimiter}在约定的交费期间内每月支付一次保险费的交费方式。
entity{tuple_delimiter}基本保险金额{tuple_delimiter}保险概念{tuple_delimiter}由投保人与保险公司约定并在保险单上载明的金额，首个保单年度的有效保险金额等于此金额。
entity{tuple_delimiter}有效保险金额{tuple_delimiter}保险概念{tuple_delimiter}用于计算保险责任的金额，首年等于基本保险金额，从第二年起每年在上一年基础上增长3.0%。
entity{tuple_delimiter}3.0%{tuple_delimiter}数据{tuple_delimiter}有效保险金额从第二个保单年度起的年增长率。
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}提供产品{tuple_delimiter}利安人寿保险股份有限公司是“利安传家鑫享终身寿险”的承保公司。
relation{tuple_delimiter}投保人{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}订立合同{tuple_delimiter}投保人与利安人寿保险股份有限公司订立保险合同。
relation{tuple_delimiter}被保险人{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}受保障{tuple_delimiter}被保险人是“利安传家鑫享终身寿险”的保障对象。
relation{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}一、产品描述{tuple_delimiter}包含章节{tuple_delimiter}产品说明书包含“产品描述”章节。
relation{tuple_delimiter}一、产品描述{tuple_delimiter}投保年龄{tuple_delimiter}定义{tuple_delimiter}“产品描述”章节定义了“投保年龄”。
relation{tuple_delimiter}投保年龄{tuple_delimiter}出生满28天至70周岁{tuple_delimiter}范围是{tuple_delimiter}投保年龄的具体范围是出生满28天至70周岁。
relation{tuple_delimiter}一、产品描述{tuple_delimiter}保险期间{tuple_delimiter}定义{tuple_delimiter}“产品描述”章节定义了“保险期间”。
relation{tuple_delimiter}保险期间{tuple_delimiter}终身{tuple_delimiter}具体为{tuple_delimiter}保险期间的具体值为终身。
relation{tuple_delimiter}一、产品描述{tuple_delimiter}交费方式{tuple_delimiter}定义{tuple_delimiter}“产品描述”章节定义了“交费方式”。
relation{tuple_delimiter}交费方式{tuple_delimiter}趸交{tuple_delimiter}包含{tuple_delimiter}交费方式包括“趸交”。
relation{tuple_delimiter}交费方式{tuple_delimiter}限期年交{tuple_delimiter}包含{tuple_delimiter}交费方式包括“限期年交”。
relation{tuple_delimiter}交费方式{tuple_delimiter}限期月交{tuple_delimiter}包含{tuple_delimiter}交费方式包括“限期月交”。
relation{tuple_delimiter}有效保险金额{tuple_delimiter}基本保险金额{tuple_delimiter}首年等于{tuple_delimiter}首个保单年度的有效保险金额等于基本保险金额。
relation{tuple_delimiter}有效保险金额{tuple_delimiter}3.0%{tuple_delimiter}年增长率{tuple_delimiter}从第二个保单年度起，有效保险金额的年增长率为3.0%。
{completion_delimiter}
""",
    """<输入文本>
```
# 利安人寿保险股份有限公司

# 利安传家鑫享终身寿险

# 产品说明书

为方便您了解和购买本保险，请您仔细阅读本产品说明书。在本产品说明书中，“您”指投保人，“我们”、“本公司”均指利安人寿保险股份有限公司。

# 三、犹豫期及退保

# 1\. 犹豫期

自您签收合同次日起，有15日的犹豫期。在此期间，请您认真审视合同，如果您认为合同与您的需求不相符，您可以在此期间提出解除合同，我们将退还您所支付的保险费。解除合同时，您需要填写申请书，并提供您的保险合同及有效身份证件。自我们收到您解除合同的书面申请时起，合同即被解除，对于合同解除前发生的保险事故，我们不承担保险责任。

# 2\. 退保

您在犹豫期后要求解除合同的，本公司自收到解除合同申请书之日起30日内向您退还保险单的现金价值。现金价值指合同保险单所具有的价值，通常体现为解除合同时，根据精算原理计算的，由本公司退还的那部分金额，保险单的现金价值将在合同中载明。您犹豫期后解除合同会遭受一定损失。
```

<输出>
entity{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}机构{tuple_delimiter}本产品说明书的发布方，在文中被称为“我们”或“本公司”。
entity{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}保险产品{tuple_delimiter}本产品说明书所介绍的保险产品。
entity{tuple_delimiter}投保人{tuple_delimiter}角色{tuple_delimiter}在文中被称为“您”，有权在犹豫期内或犹豫期后解除合同。
entity{tuple_delimiter}三、犹豫期及退保{tuple_delimiter}保险条款{tuple_delimiter}产品说明书的章节标题，说明犹豫期和退保规则。
entity{tuple_delimiter}1. 犹豫期{tuple_delimiter}保险条款{tuple_delimiter}犹豫期及退保下的子章节标题，说明犹豫期的规则。
entity{tuple_delimiter}犹豫期{tuple_delimiter}保险概念{tuple_delimiter}投保人签收合同次日起的15日内，可以无损失解除合同的期间。
entity{tuple_delimiter}15日{tuple_delimiter}期限{tuple_delimiter}犹豫期的持续时间。
entity{tuple_delimiter}保险费{tuple_delimiter}保险概念{tuple_delimiter}在犹豫期内解除合同，将由保险公司退还。
entity{tuple_delimiter}保险责任{tuple_delimiter}保险责任{tuple_delimiter}保险公司在合同解除前发生的保险事故不承担此责任。
entity{tuple_delimiter}2. 退保{tuple_delimiter}保险条款{tuple_delimiter}犹豫期及退保下的子章节标题，说明犹豫期后解除合同的规则。
entity{tuple_delimiter}退保{tuple_delimiter}保险概念{tuple_delimiter}投保人在犹豫期后解除合同的行为，会导致一定损失。
entity{tuple_delimiter}30日{tuple_delimiter}期限{tuple_delimiter}保险公司在收到退保申请后退还现金价值的期限。
entity{tuple_delimiter}现金价值{tuple_delimiter}保险概念{tuple_delimiter}在犹豫期后退保时，由保险公司退还给投保人的金额。
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}提供产品{tuple_delimiter}利安人寿保险股份有限公司是“利安传家鑫享终身寿险”的承保公司。
relation{tuple_delimiter}利安传家鑫享终身寿险{tuple_delimiter}三、犹豫期及退保{tuple_delimiter}包含章节{tuple_delimiter}产品说明书包含“犹豫期及退保”章节。
relation{tuple_delimiter}三、犹豫期及退保{tuple_delimiter}1. 犹豫期{tuple_delimiter}包含子章节{tuple_delimiter}“犹豫期及退保”章节包含“犹豫期”子章节。
relation{tuple_delimiter}三、犹豫期及退保{tuple_delimiter}2. 退保{tuple_delimiter}包含子章节{tuple_delimiter}“犹豫期及退保”章节包含“退保”子章节。
relation{tuple_delimiter}1. 犹豫期{tuple_delimiter}犹豫期{tuple_delimiter}条款定义{tuple_delimiter}该条款定义了“犹豫期”的规则。
relation{tuple_delimiter}犹豫期{tuple_delimiter}15日{tuple_delimiter}期限为{tuple_delimiter}犹豫期的持续时间为15日。
relation{tuple_delimiter}投保人{tuple_delimiter}犹豫期{tuple_delimiter}行使权利{tuple_delimiter}投保人可以在“犹豫期”内提出解除合同。
relation{tuple_delimiter}投保人{tuple_delimiter}保险费{tuple_delimiter}获得退款{tuple_delimiter}在犹豫期解除合同，投保人将获得“保险费”退款。
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}保险责任{tuple_delimiter}不承担{tuple_delimiter}对于合同解除前发生的保险事故，利安人寿保险股份有限公司不承担“保险责任”。
relation{tuple_delimiter}2. 退保{tuple_delimiter}退保{tuple_delimiter}条款定义{tuple_delimiter}该条款定义了“退保”（犹豫期后解除合同）的规则。
relation{tuple_delimiter}投保人{tuple_delimiter}退保{tuple_delimiter}发起{tuple_delimiter}投保人有权在犹豫期后发起“退保”。
relation{tuple_delimiter}投保人{tuple_delimiter}现金价值{tuple_delimiter}获得退款{tuple_delimiter}在犹豫期后退保，投保人将获得“现金价值”退款。
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}现金价值{tuple_delimiter}退还{tuple_delimiter}利安人寿保险股份有限公司在收到申请后30日内退还“现金价值”。
relation{tuple_delimiter}退保{tuple_delimiter}30日{tuple_delimiter}处理期限{tuple_delimiter}保险公司收到退保申请后的处理期限为30日。
{completion_delimiter}
""",
]

# 表格抽取示例
PROMPTS["table_extraction_examples"] = [
    """<输入文本>
```
[SOURCE:__TABLE_ENTITY_3__]
[CONTEXT]利安人寿保险股份有限公司

# 利安安康福（惠享版）重大疾病保险

# 费率表

（每万元基本保险金额对应的年交保险费）

单位：人民币元  

__TABLE_ENTITY_1__

注：月交保险费=年交保险费*0.09。

单位：人民币元  

__TABLE_ENTITY_2__

注：月交保险费=年交保险费*0.09。

单位：人民币元

__TABLE_ENTITY_3__

注：月交保险费=年交保险费*0.09。

单位：人民币元
[HTML_TABLE]
<table><tr><td>保险期间</td><td colspan="4">终身</td></tr><tr><td>身故保险金的给付方式</td><td colspan="4">方案一</td></tr><tr><td>性别</td><td colspan="2">男性</td><td colspan="2">女性</td></tr><tr><td>交费期间 年龄</td><td>5年交</td><td>10年交</td><td>5年交</td><td>10年交</td></tr><tr><td>0</td><td>408</td><td>206</td><td>393</td><td>199</td></tr><tr><td>20</td><td>717</td><td>363</td><td>687</td><td>348</td></tr><tr><td>40</td><td>1257</td><td>643</td><td>1161</td><td>596</td></tr></table>
```

<输出>
entity{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}数据型表格{tuple_delimiter}展示了“利安安康福（惠享版）”在“方案一”下，不同年龄、性别和交费期间对应的年交保险费。费率基于“每万元基本保险金额”，单位为人民币元
entity{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}机构{tuple_delimiter}上下文提及的保险产品提供方
entity{tuple_delimiter}利安安康福（惠享版）重大疾病保险{tuple_delimiter}保险产品{tuple_delimiter}本费率表所属的重大疾病保险产品
entity{tuple_delimiter}每万元基本保险金额{tuple_delimiter}保险概念{tuple_delimiter}上下文中定义的保费计算基准，表中数值均对应此基准
entity{tuple_delimiter}保险期间{tuple_delimiter}保险概念{tuple_delimiter}保险的保障期限，表中固定为“终身”
entity{tuple_delimiter}终身{tuple_delimiter}期限{tuple_delimiter}保险期间的具体值
entity{tuple_delimiter}身故保险金的给付方式{tuple_delimiter}保险概念{tuple_delimiter}身故保险金的给付方案，表中固定为“方案一”
entity{tuple_delimiter}年龄{tuple_delimiter}保险概念{tuple_delimiter}投保人年龄，作为保费计算的核心维度之一
entity{tuple_delimiter}性别{tuple_delimiter}保险概念{tuple_delimiter}投保人性别（男性/女性），作为保费计算的核心维度之一
entity{tuple_delimiter}男性{tuple_delimiter}保险概念{tuple_delimiter}作为保费计算维度的性别分类
entity{tuple_delimiter}女性{tuple_delimiter}保险概念{tuple_delimiter}作为保费计算维度的性别分类
entity{tuple_delimiter}交费期间{tuple_delimiter}保险概念{tuple_delimiter}保费的缴纳年限（如5年交, 10年交），作为保费计算的核心维度之一
entity{tuple_delimiter}5年交{tuple_delimiter}期限{tuple_delimiter}交费期间的具体值
entity{tuple_delimiter}10年交{tuple_delimiter}期限{tuple_delimiter}交费期间的具体值
entity{tuple_delimiter}年交保险费{tuple_delimiter}保险概念{tuple_delimiter}每年需要缴纳的保险费用，是本表的核心数据
entity{tuple_delimiter}月交保险费{tuple_delimiter}保险概念{tuple_delimiter}每月需要缴纳的保险费用，在上下文中定义了与年交保费的换算关系
entity{tuple_delimiter}0岁-男性-5年交-408元{tuple_delimiter}数据{tuple_delimiter}代表性数据行：0周岁男性，5年交，年交保费408元
entity{tuple_delimiter}20岁-女性-10年交-348元{tuple_delimiter}数据{tuple_delimiter}代表性数据行：20周岁女性，10年交，年交保费348元
entity{tuple_delimiter}40岁-男性-5年交-1257元{tuple_delimiter}数据{tuple_delimiter}代表性数据行：40周岁男性，5年交，年交保费1257元
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}利安安康福（惠享版）重大疾病保险{tuple_delimiter}提供产品{tuple_delimiter}利安人寿保险股份有限公司是利安安康福（惠享版）重大疾病保险的承保公司
relation{tuple_delimiter}利安安康福（惠享版）重大疾病保险{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}所属产品{tuple_delimiter}该费率表是“利安安康福（惠享版）重大疾病保险”的产品费率表
relation{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}年交保险费{tuple_delimiter}展示数据{tuple_delimiter}费率表的核心数据是“年交保险费”
relation{tuple_delimiter}年交保险费{tuple_delimiter}每万元基本保险金额{tuple_delimiter}基于{tuple_delimiter}根据上下文，年交保费是基于“每万元基本保险金额”计算的
relation{tuple_delimiter}年交保险费{tuple_delimiter}年龄{tuple_delimiter}受...影响{tuple_delimiter}“年龄”是影响“年交保险费”的关键因素
relation{tuple_delimiter}年交保险费{tuple_delimiter}性别{tuple_delimiter}受...影响{tuple_delimiter}“性别”是影响“年交保险费”的关键因素
relation{tuple_delimiter}年交保险费{tuple_delimiter}交费期间{tuple_delimiter}受...影响{tuple_delimiter}“交费期间”是影响“年交保险费”的关键因素
relation{tuple_delimiter}月交保险费{tuple_delimiter}年交保险费{tuple_delimiter}计算公式{tuple_delimiter}根据上下文：月交保险费 = 年交保险费 * 0.09
relation{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}0岁-男性-5年交-408元{tuple_delimiter}包含数据行{tuple_delimiter}费率表包含0周岁的代表性保费数据
relation{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}20岁-女性-10年交-348元{tuple_delimiter}包含数据行{tuple_delimiter}费率表包含20周岁的代表性保费数据
relation{tuple_delimiter}利安安康福（惠享版）重大疾病保险 费率表（方案一）{tuple_delimiter}40岁-男性-5年交-1257元{tuple_delimiter}包含数据行{tuple_delimiter}费率表包含40周岁的代表性保费数据
relation{tuple_delimiter}保险期间{tuple_delimiter}终身{tuple_delimiter}具体为{tuple_delimiter}表中的保险期间固定为终身
{completion_delimiter}
""",
    """<输入文本>
```
[SOURCE:__TABLE_ENTITY_1__]
[CONTEXT]# 利安人寿保险股份有限公司附加豁免保险费定期寿险条款
（2013年呈报中国保险监督管理委员会备案）
“附加豁免保险费定期寿险”简称“附加豁免定寿”。在本保险条款中，“您”指投保人，“我们”指利安人寿保险股份有限公司，“本附加险合同”指您与我们之间订立的“附加豁免保险费定期寿险合同”。
# 1. 您与我们订立的合同
__TABLE_ENTITY_1__
# 2. 我们提供的保障
__TABLE_ENTITY_2__
# 3. 如何申请豁免保险费
[HTML_TABLE]
<table><tr><td>1.1</td><td>合同构成</td><td>本附加险合同是您与我们约定保险权利义务关系的协议，包括本保险条款、保险单、投保单及其他投保文件、合法有效的声明、批注、批单及其他您与我们共同签订的书面协议。</td></tr><tr><td>1.2</td><td>合同成立与生效</td><td>您提出保险申请、我们同意承保，本附加险合同成立。本附加险合同自我们同意承保、收取首期保险费并签发保险单开始生效，具体生效日以保险单所载的日期为准。合同生效日对应日、保险费约定支付日以该日期计算。</td></tr><tr><td>1.3</td><td>投保范围</td><td>若您不是主险合同的被保险人，并分期支付主险合同的保险费，且在投保本附加险合同时您的年龄在18周岁至50周岁之间，则您可作为本附加险合同的投保人和被保险人。</td></tr><tr><td>1.4</td><td>犹豫期</td><td>自您签收本附加险合同次日起，有10日的犹豫期。在此期间，请您认真审视本附加险合同，如果您认为本附加险合同与您的需求不相符，您可以在此期间提出解除本附加险合同，我们将退还您所支付的保险费。解除合同时，您需要填写申请书，并提供您的保险合同及有效身份证件。自我们收到您解除合同的书面申请时起，本附加险合同即被解除，对于合同解除前发生的保险事故，我们不承担保险责任。</td></tr></table>
```

<输出>
entity{tuple_delimiter}附加豁免保险费定期寿险_合同订立条款表{tuple_delimiter}概念型表格{tuple_delimiter}展示了“附加豁免保险费定期寿险”合同订立的相关条款。
entity{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}机构{tuple_delimiter}保险产品的承保公司
entity{tuple_delimiter}附加豁免保险费定期寿险{tuple_delimiter}保险产品{tuple_delimiter}本条款所属的保险产品
entity{tuple_delimiter}附加豁免定寿{tuple_delimiter}保险产品{tuple_delimiter}“附加豁免保险费定期寿险”的简称
entity{tuple_delimiter}投保人{tuple_delimiter}角色{tuple_delimiter}保险合同的申请人，上下文中的“您”
entity{tuple_delimiter}条款1.1_合同构成{tuple_delimiter}保险条款{tuple_delimiter}附加豁免定寿合同构成条款的编号及标题
entity{tuple_delimiter}条款1.2_合同成立与生效{tuple_delimiter}保险条款{tuple_delimiter}附加豁免定寿合同构成条款的编号及标题
entity{tuple_delimiter}条款1.3_投保范围{tuple_delimiter}保险条款{tuple_delimiter}附加豁免定寿合同构成条款的编号及标题
entity{tuple_delimiter}条款1.4_犹豫期{tuple_delimiter}保险条款{tuple_delimiter}附加豁免定寿合同构成条款的编号及标题
entity{tuple_delimiter}犹豫期{tuple_delimiter}保险概念{tuple_delimiter}条款1.4的核心概念，即投保人解除合同的权利
entity{tuple_delimiter}解除合同的权利{tuple_delimiter}保险概念{tuple_delimiter}犹豫期内投保人享有的权利
entity{tuple_delimiter}10日{tuple_delimiter}期限{tuple_delimiter}附加豁免定寿合同犹豫期的持续时间
relation{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}附加豁免保险费定期寿险{tuple_delimiter}提供产品{tuple_delimiter}利安人寿是该产品的承保公司
relation{tuple_delimiter}附加豁免保险费定期寿险{tuple_delimiter}附加豁免定寿{tuple_delimiter}简称为{tuple_delimiter}产品简称
relation{tuple_delimiter}附加豁免保险费定期寿险_合同订立条款表{tuple_delimiter}条款1.1_合同构成{tuple_delimiter}包含条款{tuple_delimiter}表格包含条款1.1
relation{tuple_delimiter}附加豁免保险费定期寿险_合同订立条款表{tuple_delimiter}条款1.2_合同成立与生效{tuple_delimiter}包含条款{tuple_delimiter}表格包含条款1.2
relation{tuple_delimiter}附加豁免保险费定期寿险_合同订立条款表{tuple_delimiter}条款1.3_投保范围{tuple_delimiter}包含条款{tuple_delimiter}表格包含条款1.3
relation{tuple_delimiter}附加豁免保险费定期寿险_合同订立条款表{tuple_delimiter}条款1.4_犹豫期{tuple_delimiter}包含条款{tuple_delimiter}表格包含条款1.4
relation{tuple_delimiter}犹豫期{tuple_delimiter}解除合同的权利{tuple_delimiter}包含{tuple_delimiter}犹豫期包含“解除合同的权利”
relation{tuple_delimiter}犹豫期{tuple_delimiter}10日{tuple_delimiter}期限为{tuple_delimiter}犹豫期的期限为10日
{completion_delimiter}
""",
]

# 保留原有的entity_extraction_examples以确保向后兼容
PROMPTS["entity_extraction_examples"] = PROMPTS["text_extraction_examples"] + PROMPTS["table_extraction_examples"]

PROMPTS["summarize_entity_descriptions"] = """---角色---
你是一名知识图谱专家，擅长数据整理与综合归纳。

---任务---
请将给定实体或关系的多个描述合成为一份全面、连贯的总结性说明。

---指令---
1. 输入格式：描述列表以JSON格式提供，每行为一个独立的描述对象，均位于“描述列表”部分内。
2. 输出格式：合并后的说明应直接输出为多段连续文本，不添加任何额外的格式、注释或说明。
3. 全面性：总结需整合*每一条*描述中的所有关键信息，不遗漏任何重要事实或细节。
4. 上下文与客观性：
   - 总结须采用客观、第三人称的表述方式。
   - 为确保语境清晰，需在开头明确指出该实体或关系的全名。
5. 冲突处理：
   - 若描述间存在冲突或不一致，首先判断是否源自同名但不同实体/关系。
   - 如确认为不同实体/关系，请在总输出中分别进行说明。
   - 如为同一对象但存在矛盾（如历史分歧），应尽量协调并整合不同观点，或注明存在不确定性。
6. 长度限制：总说明不超过{summary_length}个token，但需保证内容深入完整。
7. 语言规范：整个输出内容必须使用{language}撰写。专有名词（如人名、地名、机构名等）如无通用正式译名或译名易造成歧义，可保持原文。

---输入---
{description_type}名称: {description_name}

描述列表:

```
{description_list}
```

---输出---
"""

PROMPTS["fail_response"] = (
    "抱歉，我无法为这个问题提供答案。[no-context]"
)

PROMPTS["rag_response"] = """---角色---

您是一位专门从提供知识库中综合信息的专家 AI 助手。您的主要功能是仅利用所提供 **上下文** 中的信息准确回答用户查询。

---目标---

生成一份全面、结构良好的答案以回应用户查询。
答案必须整合 **上下文** 中的知识图谱数据与文档片段。
如有对话历史，请考虑以保持对话连贯并避免重复信息。

---说明---

1. 步骤说明：
  - 在对话历史上下文中，仔细分析用户的查询意图，全面理解其信息需求。
  - 严格审查**Context**中的`知识图谱数据`与`文档片段`，识别并提取与回答用户问题直接相关的全部信息。
  - 将提取到的事实融入流畅且逻辑连贯的回复中。只能利用你自己的知识对句子进行润色和衔接，不得引入任何外部信息或知识。
  - 跟踪直接支撑回复内容的文档片段的reference_id，并将reference_id与`参考文献列表`中的条目关联，以生成正确的文献引用。
  - 在回答末尾生成一个参考文献部分。每一条参考文献都必须直接支撑回答中的事实。
  - 参考文献部分之后，不得输出任何内容。

2. 内容要求与事实依据：
  - 必须严格限定在**Context**（上下文）中提供的信息回答，绝不能凭空编造、假设或推断上下文未明确指出的信息。
  - 如果**Context**不足以得出答案，请明确说明信息不足，切勿尝试猜测。

3. 格式和语言要求：
  - 回复内容**必须**与用户提问的语言一致。
  - 回复**必须**采用Markdown格式，提升清晰度和结构化（如标题、加粗文本、项目符号等）。
  - 回复应以{response_type}形式展现。

4. 参考文献部分格式：
  - 参考文献部分标题为：`### References`
  - 参考文献列表每条格式为：`- [n] Document Title`。方括号`[`后面不应出现插入符号`^`。
  - 引用中的Document Title应保持其原有语言。
  - 每条引用单独成行。
  - 最多列出5条最相关的参考文献。
  - 不得生成脚注部分，或在参考文献后添加任何注释、总结或说明。

5. 参考文献部分示例：
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. 额外说明：{user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """--角色---
你是一名保险领域的知识图谱检索专家。你的任务是为检索增强生成（RAG）系统解析用户查询，提取用于在知识图谱中分别检索**实体（节点）**和**关系（边）**的关键词。

---目标---
根据用户查询，提取两种目的明确的关键词：

1.  **high_level_keywords (高层关键词)**：
    *   **用途**：专门用于检索知识图谱中的**关系（边）**。
    *   **提取内容**：捕捉查询中隐含的**宏观概念、主题、或实体间的互动关系**。这些词通常描述一个过程（如“理赔流程”）、一个属性（如“现金价值计算”）、或一个比较（如“区别”）。它们回答了“实体之间发生了什么？”或“实体有什么样的属性或联系？”。

2.  **low_level_keywords (低层关键词)**：
    *   **用途**：专门用于检索知识图谱中的**实体（节点）**。
    *   **提取内容**：识别查询中提到的**具体实体、专有名词、技术术语或关键细节**。这包括保险产品名、机构名、具体的条款编号及标题、日期、金额、年龄等。它们回答了“查询涉及哪些具体的人、事、物？”。

---说明与约束---
1.  **输出格式**：你的输出必须是一个有效的JSON对象，不能包含其他任何内容。不要包含任何解释性文本、Markdown代码围栏（如```json），或JSON前后的任何其他文本。它将由JSON解析器直接解析。
2.  **检索导向**：提取关键词时，必须时刻思考它们将如何被用于检索。`high_level` 关键词应能匹配到关系/边的描述，而 `low_level` 关键词应能直接命中实体/节点的名称或属性。
3.  **领域知识应用**：当用户使用口语化表达时，利用你的保险领域知识将其映射为标准的图谱概念。例如，用户问“出事了怎么赔？”，`high_level` 关键词应包含“理赔流程”、“保险责任”；用户问“这款产品谁能买？”，`high_level` 关键词应包含“投保条件”、“投保范围”。
4.  **简洁且有意义**：关键词应为简洁的词语或有意义的短语。当多词短语代表单一概念时，应优先考虑。
5.  **处理边缘情况**：对于过于简单、模糊或无意义的查询（例如，“你好”、“好的”、“asdfghjkl”），你必须返回一个JSON对象，其中两种关键词类型的列表均为空。

---示例---
{examples}

---实际数据---
用户查询：{query}

---输出---
输出"""


PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "传家鑫享和传家福这两款终身寿险有什么区别？"

Output:
{
  "high_level_keywords": ["区别", "产品比较", "差异"],
  "low_level_keywords": ["传家鑫享", "传家福", "终身寿险"]
}

""",
    """Example 2:

Query: "如果被保险人身故，如何申请理赔？需要哪些材料？"

Output:
{
  "high_level_keywords": ["身故理赔", "理赔申请", "申请材料", "理赔流程"],
  "low_level_keywords": ["被保险人"]
}

""",
    """Example 3:

Query: "什么是现金价值？它和保单贷款有什么关系？"

Output:
{
  "high_level_keywords": ["定义", "关系"],
  "low_level_keywords": ["现金价值", "保单贷款"]
}

""",
    """Example 4:

Query: "保险条款里提到的‘不可抗辩条款’是什么意思？"

Output:
{
  "high_level_keywords": ["定义", "条款释义"],
  "low_level_keywords": ["不可抗辩条款"]
}

""",
    """Example 5:

Query: "酒后驾驶导致的意外，在利安百万身价两全保险的责任免除里吗？"

Output:
{
  "high_level_keywords": ["责任免除", "免责条款", "保障范围"],
  "low_level_keywords": ["酒后驾驶", "利安百万身价两全保险"]
}

""",
]
