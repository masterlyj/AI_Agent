from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# 所有分隔符的格式必须为“<|大写字符串|>”
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---角色说明---
你是一名知识图谱专家，负责从输入文本中抽取实体和关系。

---操作指南---
1.  **实体抽取与输出：**
    *   **识别：** 识别输入文本中明确定义且有意义的实体。
    *   **实体信息：** 针对每个被识别的实体，提取以下信息：
        *   `entity_name`：实体名称。若实体名称不区分大小写，请将每个重要词汇首字母大写（标题式大小写），并在整个抽取过程中保持**命名一致性**。
        *   `entity_type`：将实体归类为以下类型之一：{entity_types}。若不属于这些类型中的任何一种，不要添加新类别，请标记为 `Other`。
        *   `entity_description`：仅根据输入文本，简明、全面地描述实体的属性和活动。
    *   **实体输出格式：** 每个实体共输出4个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条实体。第一个字段必须是字面字符串 `entity`。
        *   格式：`entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **关系抽取与输出：**
    *   **识别：** 识别已抽取实体之间直接、明确且有意义的关系。
    *   **N元关系拆分：** 若一句话涉及超过两个实体的关系（即N元关系），请将其拆分为多个二元（两个实体之间的）关系分别描述。
        *   **举例：** 若文本为“Alice、Bob 与 Carol 共同参与了 Project X”，请抽取如“Alice 与 Project X 合作”、“Bob 与 Project X 合作”、“Carol 与 Project X 合作”，或者“Alice 与 Bob 合作”，根据最合理的二元关系理解输出。
    *   **关系信息：** 对于每个二元关系，提取以下字段：
        *   `source_entity`：关系的起始实体，命名需与实体抽取部分**保持一致**。若名称不区分大小写，按标题式大小写输出。
        *   `target_entity`：关系的目标实体，命名需与实体抽取部分**保持一致**。若名称不区分大小写，按标题式大小写输出。
        *   `relationship_keywords`：一个或多个用于概括关系本质、主题或概念的关键词。多个关键词请用中文逗号 `,` 分隔。**请勿用 `{tuple_delimiter}` 分隔关键词。**
        *   `relationship_description`：简明说明该二元关系本质，为实体间的关联提供明确理由。
    *   **关系输出格式：** 每个关系共输出5个字段，使用 `{tuple_delimiter}` 分隔，同一行为一条关系。第一个字段必须是字面字符串 `relation`。
        *   格式：`relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **分隔符使用规范：**
    *   `{tuple_delimiter}` 是一个完整且不可被填充的分隔标记，仅作为字段分隔符来使用。
    *   **错误示例：** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **正确示例：** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **关系方向与去重：**
    *   所有关系默认为**无向关系**（除非明确指定有向）。交换源实体和目标实体的顺序不会被视为新关系。
    *   请避免输出重复关系。

5.  **输出顺序与优先级：**
    *   先输出所有抽取的实体，后输出所有关系。
    *   在关系列表中，将**与输入文本核心意义最相关**的关系优先输出。

6.  **客观与上下文要求：**
    *   所有实体名称及描述须使用**第三人称**书写。
    *   必须明确指出主体或客体；**避免使用**如“本文”、“本公司”、“我们”、“你”、“他/她”等指代性人称。

7.  **语言与专有名词：**
    *   整个输出（实体名称、关键词、描述）必须使用 `{language}`。
    *   专有名词（如人名、地名、组织名等）如果没有公认翻译或翻译会造成歧义，请保留原文。

8.  **输出终止标志：** 所有实体与关系输出完毕且完全满足以上要求后，最后一行仅输出字面字符串 `{completion_delimiter}` 作为终止信号。

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

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract comprehensive insurance-specific entities and relationships from the input text, which may include policy clauses, rate tables, product descriptions, regulatory terms, and procedural documents.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.

2.  **Comprehensive Insurance Entity Types:** Prioritize the extraction of the following entity types commonly found in insurance documents:
    - `InsuranceCompany` (e.g., "利安人寿保险股份有限公司")
    - `InsuranceProduct` (e.g., "附加豁免保险费定期寿险", "传家宝终身寿险", "传家福终身寿险")
    - `Clause` (e.g., "保险责任", "责任免除", "投保范围", "犹豫期")
    - `BenefitType` (e.g., "身故保险金", "豁免保险费", "现金价值", "生存金")
    - `ExclusionCondition` (e.g., "遗传性疾病", "先天性畸形", "既往症", "猝死")
    - `PolicyTerm` (e.g., "保险期间", "犹豫期", "宽限期", "交费期间")
    - `RateTable` (e.g., "费率表", "定价表", "精算表")
    - `AgeRange` (e.g., "18周岁至50周岁", "出生满28天至70周岁")
    - `Gender` (e.g., "男性", "女性")
    - `TimePeriod` (e.g., "10日", "180日", "2年内", "终身")
    - `MonetaryAmount` (e.g., "保险费", "保险金额", "现金价值", "保费")
    - `Percentage` (e.g., "3.5%", "160%", "80%")
    - `Definition` (e.g., "意外伤害", "全残", "周岁", "有效身份证件")
    - `Procedure` (e.g., "保单质押贷款", "减保", "解除合同", "转换年金")
    - `RegulatoryReference` (e.g., "中国保险监督管理委员会", "备案")
    - `ClauseNumber` (e.g., "条款1.4", "2.3", "第3条")
    - `DocumentSection` (e.g., "投保范围", "保险责任", "如何申请保险金")

3.  **Structured Data Processing:** 
    - **Rate Tables**: Extract each rate entry as a structured entity with relationships to age, gender, term, and monetary values
    - **Clause Numbers**: Extract clause references with their hierarchical relationships (e.g., "1.1 合同构成" → "合同构成" entity with clause number "1.1")
    - **Tables and Lists**: Extract each row/entry as separate entities with appropriate relationships
    - **Definition Lists**: Extract each definition with its explanatory content

4.  **Insurance-Specific Relationship Types:**
    - **Product Relationships**: Product-to-company, product-to-clause, product-to-rate-table
    - **Clause Relationships**: Clause-to-definition, clause-to-procedure, clause-to-condition
    - **Benefit Relationships**: Benefit-to-product, benefit-to-condition, benefit-to-procedure
    - **Temporal Relationships**: Term-to-product, period-to-condition, age-to-rate
    - **Regulatory Relationships**: Product-to-regulator, clause-to-regulation
    - **Hierarchical Relationships**: Main-clause-to-sub-clause, section-to-subsection

5.  **Cross-Document Linking:** If multiple documents are provided, identify relationships that link entities across documents:
    - Product names appearing in both clause documents and rate tables
    - Related terms and definitions spanning different product documents
    - Company references across various insurance products
    - Regulatory references and compliance requirements

6.  **Numerical and Temporal Precision:**
    - Extract exact age ranges, time periods, and monetary amounts
    - Preserve percentage values and rate calculations
    - Identify specific dates, deadlines, and effective periods
    - Capture conditional timeframes (e.g., "180日后", "2年内")

7.  **Definition and Terminology Extraction:**
    - Extract all insurance term definitions with their complete explanations
    - Identify technical jargon and its contextual meaning
    - Capture procedural definitions and their application conditions
    - Extract regulatory terminology and compliance requirements

8.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.

9.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.

10. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
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

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
利安人寿保险股份有限公司传家宝终身寿险条款
"传家宝终身寿险"简称"传家宝"。在本保险条款中，"您"指投保人，"我们"指利安人寿保险股份有限公司，"本合同"指您与我们之间订立的"传家宝终身寿险合同"。

1.3 投保年龄
指您投保时被保险人的年龄，本合同接受的被保险人的投保年龄范围为出生满28天至70周岁。

2.1 保险金额
本合同的基本保险金额由您在投保时与我们约定并在保险单中载明。从第二个保单年度起，各保单年度的有效保险金额计算公式如下：
第n个保单年度的有效保险金额 = 基本保险金额 × 1.035^(n-1)
```

<Output>
entity{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}InsuranceCompany{tuple_delimiter}利安人寿保险股份有限公司是一家提供人寿保险服务的保险公司
entity{tuple_delimiter}传家宝终身寿险{tuple_delimiter}InsuranceProduct{tuple_delimiter}传家宝终身寿险是利安人寿推出的终身寿险产品
entity{tuple_delimiter}传家宝{tuple_delimiter}InsuranceProduct{tuple_delimiter}传家宝是传家宝终身寿险的简称
entity{tuple_delimiter}投保年龄{tuple_delimiter}Clause{tuple_delimiter}规定被保险人的年龄要求
entity{tuple_delimiter}保险金额{tuple_delimiter}Clause{tuple_delimiter}规定保险金额的计算和确定方式
entity{tuple_delimiter}基本保险金额{tuple_delimiter}MonetaryAmount{tuple_delimiter}保险合同约定的基本保险金额
entity{tuple_delimiter}有效保险金额{tuple_delimiter}MonetaryAmount{tuple_delimiter}按公式计算的有效保险金额
entity{tuple_delimiter}出生满28天至70周岁{tuple_delimiter}AgeRange{tuple_delimiter}投保年龄范围为出生满28天至70周岁
entity{tuple_delimiter}1.035{tuple_delimiter}Percentage{tuple_delimiter}有效保险金额计算公式中的增长率系数
entity{tuple_delimiter}条款1.3{tuple_delimiter}ClauseNumber{tuple_delimiter}投保年龄条款的编号
entity{tuple_delimiter}条款2.1{tuple_delimiter}ClauseNumber{tuple_delimiter}保险金额条款的编号
relation{tuple_delimiter}传家宝终身寿险{tuple_delimiter}利安人寿保险股份有限公司{tuple_delimiter}产品提供，公司关系{tuple_delimiter}传家宝终身寿险由利安人寿保险股份有限公司提供
relation{tuple_delimiter}传家宝{tuple_delimiter}传家宝终身寿险{tuple_delimiter}简称关系，产品关联{tuple_delimiter}传家宝是传家宝终身寿险的简称
relation{tuple_delimiter}投保年龄{tuple_delimiter}出生满28天至70周岁{tuple_delimiter}年龄限制，资格条件{tuple_delimiter}投保年龄规定被保险人的年龄范围为出生满28天至70周岁
relation{tuple_delimiter}保险金额{tuple_delimiter}基本保险金额{tuple_delimiter}金额类型，包含关系{tuple_delimiter}保险金额包含基本保险金额
relation{tuple_delimiter}有效保险金额{tuple_delimiter}1.035{tuple_delimiter}计算公式，增长系数{tuple_delimiter}有效保险金额按1.035的增长率计算
{completion_delimiter}
""",
    """<Input Text>
```
附加豁免保险费定期寿险费率表
(每豁免所保障的保险合同的1000元年交保险费对应的费率)

年龄 | 性别：男性 | 单位：人民币元
| 1 | 2 | 3 | 4 | 5
18 | 0.70 | 1.09 | 1.51 | 1.95 | 2.40
19 | 0.78 | 1.20 | 1.65 | 2.12 | 2.60
20 | 0.84 | 1.30 | 1.77 | 2.26 | 2.76

1.3 投保范围
若您不是主险合同的被保险人,并分期支付主险合同的保险费,且在投保本附加险合同时您的年龄在18周岁至50周岁之间,则您可作为本附加险合同的投保人和被保险人。
```

<Output>
entity{tuple_delimiter}附加豁免保险费定期寿险{tuple_delimiter}InsuranceProduct{tuple_delimiter}附加豁免保险费定期寿险是一种附加险产品
entity{tuple_delimiter}附加豁免保险费定期寿险费率表{tuple_delimiter}RateTable{tuple_delimiter}附加豁免保险费定期寿险的费率表，按年龄、性别和保险期间列示费率
entity{tuple_delimiter}投保范围{tuple_delimiter}Clause{tuple_delimiter}规定投保人的资格条件和年龄限制
entity{tuple_delimiter}18周岁{tuple_delimiter}Age{tuple_delimiter}费率表中的最低年龄
entity{tuple_delimiter}20周岁{tuple_delimiter}Age{tuple_delimiter}费率表中的年龄
entity{tuple_delimiter}男性{tuple_delimiter}Gender{tuple_delimiter}费率表适用的性别类型
entity{tuple_delimiter}保险期间1年{tuple_delimiter}PolicyTerm{tuple_delimiter}保险期间为1年对应的费率
entity{tuple_delimiter}保险期间2年{tuple_delimiter}PolicyTerm{tuple_delimiter}保险期间为2年对应的费率
entity{tuple_delimiter}费率0.70{tuple_delimiter}MonetaryAmount{tuple_delimiter}18岁男性1年保险期间的费率为0.70元
entity{tuple_delimiter}费率0.84{tuple_delimiter}MonetaryAmount{tuple_delimiter}20岁男性1年保险期间的费率为0.84元
entity{tuple_delimiter}18周岁至50周岁{tuple_delimiter}AgeRange{tuple_delimiter}投保年龄范围为18周岁至50周岁
entity{tuple_delimiter}分期支付{tuple_delimiter}Procedure{tuple_delimiter}保险费支付方式为分期支付
entity{tuple_delimiter}1000元{tuple_delimiter}MonetaryAmount{tuple_delimiter}费率计算的基础金额
relation{tuple_delimiter}附加豁免保险费定期寿险{tuple_delimiter}附加豁免保险费定期寿险费率表{tuple_delimiter}产品关联，定价依据{tuple_delimiter}附加豁免保险费定期寿险有对应的费率表
relation{tuple_delimiter}附加豁免保险费定期寿险费率表{tuple_delimiter}18周岁{tuple_delimiter}适用年龄，定价因素{tuple_delimiter}费率表包含18周岁的费率数据
relation{tuple_delimiter}附加豁免保险费定期寿险费率表{tuple_delimiter}男性{tuple_delimiter}适用性别，定价因素{tuple_delimiter}费率表包含男性的费率数据
relation{tuple_delimiter}18周岁{tuple_delimiter}费率0.70{tuple_delimiter}对应费率，定价关系{tuple_delimiter}18周岁对应1年期的费率为0.70
relation{tuple_delimiter}20周岁{tuple_delimiter}费率0.84{tuple_delimiter}对应费率，定价关系{tuple_delimiter}20周岁对应1年期的费率为0.84
relation{tuple_delimiter}投保范围{tuple_delimiter}18周岁至50周岁{tuple_delimiter}年龄限制，资格条件{tuple_delimiter}投保范围规定年龄限制为18周岁至50周岁
relation{tuple_delimiter}费率表{tuple_delimiter}1000元{tuple_delimiter}计算基础，费率依据{tuple_delimiter}费率表基于1000元保险费计算
{completion_delimiter}

""",
    """<Input Text>
```
2.4 责任免除
因下列情形之一导致被保险人身故的，我们不承担给付保险金的责任：
(1) 投保人对被保险人的故意杀害、故意伤害;
(2) 被保险人故意自伤、故意犯罪或者抗拒依法采取的刑事强制措施;
(3) 被保险人自本合同成立或合同效力恢复之日起2年内自杀，但被保险人自杀时为无民事行为能力人的除外；
(4) 被保险人斗殴、吸食或注射毒品;
(5) 被保险人酒后驾驶,无合法有效驾驶证驾驶,或驾驶无有效行驶证的机动车;

7.5 意外伤害
指遭受外来的、突发的、非本意的、非疾病的客观事件直接致使身体受到的伤害。本附加险合同所述的意外伤害导致的身故，不包括猝死。

7.6 全残
本附加险合同所述"全残"是指下列情形之一:
(1) 双目永久完全失明(注1);
(2) 两上肢腕关节以上或两下肢踝关节以上缺失;
(3) 一上肢腕关节以上及一下肢踝关节以上缺失;
```

<Output>
entity{tuple_delimiter}责任免除{tuple_delimiter}Clause{tuple_delimiter}规定保险公司不承担保险责任的情形
entity{tuple_delimiter}意外伤害{tuple_delimiter}Definition{tuple_delimiter}外来的、突发的、非本意的、非疾病的客观事件导致的伤害
entity{tuple_delimiter}全残{tuple_delimiter}Definition{tuple_delimiter}完全残疾，包括双目失明、肢体缺失等八种情形
entity{tuple_delimiter}故意杀害{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，投保人对被保险人的故意杀害
entity{tuple_delimiter}故意自伤{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，被保险人故意自伤
entity{tuple_delimiter}自杀{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，合同成立后2年内自杀
entity{tuple_delimiter}斗殴{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，被保险人斗殴
entity{tuple_delimiter}吸食毒品{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，吸食或注射毒品
entity{tuple_delimiter}酒后驾驶{tuple_delimiter}ExclusionCondition{tuple_delimiter}责任免除情形之一，酒后驾驶机动车
entity{tuple_delimiter}双目永久完全失明{tuple_delimiter}DisabilityCondition{tuple_delimiter}全残情形之一，双眼永久性失明
entity{tuple_delimiter}上肢腕关节以上缺失{tuple_delimiter}DisabilityCondition{tuple_delimiter}全残情形之一，手腕以上上肢缺失
entity{tuple_delimiter}下肢踝关节以上缺失{tuple_delimiter}DisabilityCondition{tuple_delimiter}全残情形之一，脚踝以上下肢缺失
entity{tuple_delimiter}猝死{tuple_delimiter}ExclusionCondition{tuple_delimiter}意外伤害导致的身故不包括猝死
entity{tuple_delimiter}2年内{tuple_delimiter}TimePeriod{tuple_delimiter}自杀免责的时间限制
entity{tuple_delimiter}条款2.4{tuple_delimiter}ClauseNumber{tuple_delimiter}责任免除条款的编号
entity{tuple_delimiter}条款7.5{tuple_delimiter}ClauseNumber{tuple_delimiter}意外伤害定义条款的编号
entity{tuple_delimiter}条款7.6{tuple_delimiter}ClauseNumber{tuple_delimiter}全残定义条款的编号
relation{tuple_delimiter}责任免除{tuple_delimiter}故意杀害{tuple_delimiter}包含情形，免责条件{tuple_delimiter}故意杀害是责任免除的情形之一
relation{tuple_delimiter}责任免除{tuple_delimiter}故意自伤{tuple_delimiter}包含情形，免责条件{tuple_delimiter}故意自伤是责任免除的情形之一
relation{tuple_delimiter}责任免除{tuple_delimiter}自杀{tuple_delimiter}包含情形，免责条件{tuple_delimiter}2年内自杀是责任免除的情形之一
relation{tuple_delimiter}自杀{tuple_delimiter}2年内{tuple_delimiter}时间限制，免责条件{tuple_delimiter}自杀免责适用于合同成立后2年内
relation{tuple_delimiter}意外伤害{tuple_delimiter}猝死{tuple_delimiter}排除关系，除外责任{tuple_delimiter}意外伤害定义明确排除猝死
relation{tuple_delimiter}全残{tuple_delimiter}双目永久完全失明{tuple_delimiter}包含情形，定义关系{tuple_delimiter}双目失明是全残的一种情形
relation{tuple_delimiter}全残{tuple_delimiter}上肢腕关节以上缺失{tuple_delimiter}包含情形，定义关系{tuple_delimiter}上肢缺失是全残的一种情形
{completion_delimiter}
""",
    """<Input Text>
```
1.4 犹豫期
自您签收本合同次日起，有10日的犹豫期。在此期间，请您认真审视本合同，如果您认为本合同与您的需求不相符，您可以在此期间提出解除本合同，我们将退还您所支付的保险费。

5.2 保单质押贷款
在本合同有效期内，您可以申请并经我们审核同意后办理保单质押贷款。贷款金额不得超过保险单的现金价值扣除各项欠款后余额的80%，每次贷款期限最长不超过6个月，贷款利率按您与我们签订的贷款协议中约定的利率执行。

9.1 保单年度
从本合同生效日或合同生效日对应日的零时起至下一个合同生效日对应日前一日的24时止为一个保单年度。
```

<Output>
entity{tuple_delimiter}犹豫期{tuple_delimiter}PolicyTerm{tuple_delimiter}签收合同后10日内可无条件解除合同的期间
entity{tuple_delimiter}保单质押贷款{tuple_delimiter}Procedure{tuple_delimiter}以保单现金价值为质押的贷款业务
entity{tuple_delimiter}保单年度{tuple_delimiter}Definition{tuple_delimiter}从合同生效日起至下一个合同生效日对应日前一日的24时止
entity{tuple_delimiter}10日{tuple_delimiter}TimePeriod{tuple_delimiter}犹豫期的持续时间为10天
entity{tuple_delimiter}解除合同{tuple_delimiter}Procedure{tuple_delimiter}在犹豫期内取消保险合同的权利
entity{tuple_delimiter}退还保险费{tuple_delimiter}Benefit{tuple_delimiter}犹豫期内解除合同时退还已交保险费
entity{tuple_delimiter}现金价值{tuple_delimiter}MonetaryAmount{tuple_delimiter}保险单所具有的价值
entity{tuple_delimiter}80%{tuple_delimiter}Percentage{tuple_delimiter}贷款金额不得超过现金价值的80%
entity{tuple_delimiter}6个月{tuple_delimiter}TimePeriod{tuple_delimiter}每次贷款期限最长不超过6个月
entity{tuple_delimiter}贷款利率{tuple_delimiter}Percentage{tuple_delimiter}按贷款协议约定的利率执行
entity{tuple_delimiter}合同生效日{tuple_delimiter}Date{tuple_delimiter}保险合同开始生效的日期
entity{tuple_delimiter}条款1.4{tuple_delimiter}ClauseNumber{tuple_delimiter}犹豫期条款的编号
entity{tuple_delimiter}条款5.2{tuple_delimiter}ClauseNumber{tuple_delimiter}保单质押贷款条款的编号
entity{tuple_delimiter}条款9.1{tuple_delimiter}ClauseNumber{tuple_delimiter}保单年度定义条款的编号
relation{tuple_delimiter}犹豫期{tuple_delimiter}10日{tuple_delimiter}时间属性，期限规定{tuple_delimiter}犹豫期持续10天
relation{tuple_delimiter}犹豫期{tuple_delimiter}解除合同{tuple_delimiter}允许操作，客户权利{tuple_delimiter}犹豫期内可解除合同
relation{tuple_delimiter}解除合同{tuple_delimiter}退还保险费{tuple_delimiter}结果关联，资金处理{tuple_delimiter}解除合同后退还保险费
relation{tuple_delimiter}保单质押贷款{tuple_delimiter}现金价值{tuple_delimiter}质押基础，贷款依据{tuple_delimiter}保单质押贷款以现金价值为质押
relation{tuple_delimiter}保单质押贷款{tuple_delimiter}80%{tuple_delimiter}贷款限额，比例限制{tuple_delimiter}贷款金额不得超过现金价值的80%
relation{tuple_delimiter}保单质押贷款{tuple_delimiter}6个月{tuple_delimiter}期限限制，时间约束{tuple_delimiter}每次贷款期限最长不超过6个月
relation{tuple_delimiter}保单年度{tuple_delimiter}合同生效日{tuple_delimiter}时间起点，年度计算{tuple_delimiter}保单年度从合同生效日起计算
{completion_delimiter}
""",
]

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

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
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

PROMPTS["keywords_extraction"] = """---Role---
You are an expert insurance domain keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system in the insurance industry. Your purpose is to identify semantic keywords that bridge the gap between user's natural language queries and formal insurance terminology.

---Goal---
Given a user query, your task is to extract two distinct types of keywords following LightRAG's design:
1. **high_level_keywords**: Overarching concepts, themes, core intent, subject areas, or question types in insurance domain
2. **low_level_keywords**: Specific entities, proper nouns, technical jargon, product names, numerical values, or concrete details

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.

2. **Keyword Classification Guidelines**:
   - **high_level_keywords**: Extract overarching concepts, themes, core intent, subject areas, or question types. Focus on insurance domain concepts like policy types, benefit categories, procedural terms, and query intentions.
   - **low_level_keywords**: Extract specific entities, proper nouns, technical terms, product names, numerical values, time periods, demographic info, and concrete details from the query.

3. **Insurance Domain Expertise**: Prioritize extraction of:
   - **Product Names**: Full product names, abbreviated versions, and product versions (e.g., "附加豁免保险费定期寿险", "传家宝终身寿险", "2023版")
   - **Company Names**: Insurance company names (e.g., "利安人寿保险股份有限公司")
   - **Clause References**: Specific clause types and numbers (e.g., "保险责任", "责任免除", "条款2.3", "投保范围")
   - **Benefit Types**: Specific benefit categories (e.g., "身故保险金", "豁免保险费", "现金价值", "生存金")
   - **Procedural Terms**: Insurance procedures and policies (e.g., "犹豫期", "宽限期", "保单质押贷款", "减保")

4. **Semantic Bridging**: Map colloquial expressions to formal insurance terminology:
   - "快返型产品" → "生存金短期返还"
   - "养老社区保险" → "CCRC保险" 
   - "万能险" → "万能型保险"
   - "重疾险" → "重大疾病保险"
   - "年金险" → "年金保险"

5. **Structural Information Recognition**: Identify document structure references:
   - **Document Types**: "费率表", "保险条款", "释义", "产品说明书"
   - **Clause Numbers**: "条款1.4", "2.3", "第3条"
   - **Section References**: "投保范围", "保险责任", "责任免除", "如何申请保险金"

6. **Numerical & Temporal Data Extraction**: Capture:
   - **Age Ranges**: "18周岁至50周岁", "出生满28天至70周岁", "30岁", "男性"
   - **Time Periods**: "10日犹豫期", "180日等待期", "2年内"
   - **Percentages & Rates**: "3.5%", "费率表", "160%"
   - **Monetary Amounts**: "保险费", "保险金额", "现金价值"
   - **Quantities**: "8种情形", "80%", "6个月"

7. **Cross-Document Entity Linking**: Identify entities that may appear across multiple documents:
   - Product names appearing in both clause documents and rate tables
   - Related terms and concepts spanning different product documents
   - Company references across various insurance products

8. **Quality Assurance**:
   - All keywords must be explicitly present in or directly inferable from the user query
   - Avoid extracting keywords that are not relevant to the insurance domain
   - Ensure both keyword categories contain content when applicable
   - Maintain consistency in terminology extraction

9. **Edge Case Handling**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "利安人寿传家宝终身寿险的投保年龄范围是多少？"

Output:
{
  "high_level_keywords": ["投保年龄范围", "保险产品查询", "投保条件"],
  "low_level_keywords": ["利安人寿", "传家宝终身寿险", "出生满28天至70周岁"]
}

""",
    """Example 2:

Query: "附加豁免保险费定期寿险费率表中，30岁男性5年期费率是多少？"

Output:
{
  "high_level_keywords": ["费率查询", "费率表查询", "定价信息"],
  "low_level_keywords": ["附加豁免保险费定期寿险", "费率表", "30岁", "男性", "5年期"]
}

""",
    """Example 3:

Query: "传家福终身寿险的责任免除条款2.4包括哪些内容？"

Output:
{
  "high_level_keywords": ["责任免除", "免责条款", "保险责任"],
  "low_level_keywords": ["传家福终身寿险", "条款2.4", "除外责任"]
}

""",
    """Example 4:

Query: "犹豫期是多久？可以全额退保吗？"

Output:
{
  "high_level_keywords": ["犹豫期", "退保政策", "解除合同"],
  "low_level_keywords": ["10日", "15日", "全额退还", "保险费退还"]
}

""",
    """Example 5:

Query: "全残的定义包括哪些情况？双目失明算全残吗？"

Output:
{
  "high_level_keywords": ["全残定义", "残疾认定", "保险释义"],
  "low_level_keywords": ["双目失明", "8种情形", "永久完全", "肢体缺失"]
}

""",
    """Example 6:

Query: "保单质押贷款能贷多少？利率怎么算？"

Output:
{
  "high_level_keywords": ["保单质押贷款", "贷款政策", "现金价值权益"],
  "low_level_keywords": ["80%", "6个月", "现金价值", "贷款利率"]
}

""",
    """Example 7:

Query: "水陆公共交通意外身故保险金和航空意外保险金能同时获得吗？"

Output:
{
  "high_level_keywords": ["意外保险金", "保险责任", "赔偿规则"],
  "low_level_keywords": ["水陆公共交通意外", "航空意外", "2000万元", "不可兼得"]
}

""",
    """Example 8:

Query: "传家宝终身寿险的现金价值如何计算？"

Output:
{
  "high_level_keywords": ["现金价值", "价值计算", "退保价值"],
  "low_level_keywords": ["传家宝终身寿险", "精算原理", "解除合同"]
}

""",
]
