from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "Chinese"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Formula", "Variable", "Model", "Financial_Product", "Financial_Concept"]

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---目标---
给定一份关于金融工程、量化金融或衍生品定价的技术文档，请识别并提取所有关键信息实体及其之间的关系。输出应为该文档核心知识的结构化表示。
输出语言请使用 {language}。

---步骤---
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果为英文，首字母大写。
- entity_type：必须**仅**从以下列表中选择类型：[{entity_types}]。不在此列表中的类型（如“Technique”、“Method”等）均视为**无效**，不得使用。
- entity_description：仅基于输入文本中明确信息，全面描述该实体的属性和作用。**不得推断或虚构文本未明确说明的信息。**
    - 对于 **Financial_Product**，描述应说明其核心机制、收益结构和主要特征。
    - 对于 **Financial_Concept**，描述应说明其含义及相关性。
    - 对于 **Formula**，说明其用途及计算内容。
    - 对于 **Variable**，仅提取在模型或公式中代表参数、随机变量或状态变量的代数符号（如 S、r_d、σ）。变量应代表一般性概念（如“波动率”或“利率”），而非具体数值。严格排除单独的数值（如 1.6、100、0.05）和日期，这些属于数据点或常数，不是变量。
    - 对于 **Model**，仅提取文本中明确提及的数学、统计或金融建模框架。
    - 如果文本信息不足，请写“文本中未提供描述”。
每个实体格式如下：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 基于第1步识别的实体，找出所有*明确相关*的（source_entity, target_entity）实体对。
对于每对相关实体，提取以下信息：
- source_entity：源实体名称，来自第1步。
- target_entity：目标实体名称，来自第1步。
- relationship_description：说明为何认为这两个实体相关。
- relationship_strength：用数字表示两实体关系强度。
- relationship_keywords：用一个或多个高层次关键词总结关系本质。可选：[Derives_from, Defines, Based_on_assumption, Is_a_component_of, Is_an_example_of, Uses_technique, Prices, Describes_payoff_of]。
每个关系格式如下：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 总结全文的高层次关键词，概括主要概念、主题或话题。
格式如下：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 输出请用 {language}，将第1步和第2步识别的所有实体和关系作为一个列表输出。列表项之间用**{record_delimiter}**分隔。

5. 输出结束时请输出 {completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---真实数据---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    r"""Example 1:

Entity_types: [Formula, Variable, Model, Financial_Product, Financial_Concept]
Text:
```
The pricing follows the usual procedures of Arbitrage pricing theory and the Fundamental theorem of asset pricing. 

In a Foreign Exchange market this means that we model the underlying exchange rate by a geometric Brownian motion

$$
d S_{{t}} = ( r_{{d}} - r_{{f}} ) S_{{t}} d t + \sigma S_{{t}} d W_{{t}} ,
$$

where $r_{{d}}$ denotes the domestic interest rate, $\sigma$ the volatility, $W_{{t}}$ as standard Brownian motion, see Foreign Exchange symmetries for details. 
```

Output:
("entity"{tuple_delimiter}"Arbitrage Pricing Theory"{tuple_delimiter}"Financial_Concept"{tuple_delimiter}"A fundamental theory used as a procedural basis for pricing financial instruments, ensuring no risk-free profit opportunities exist."){record_delimiter}
("entity"{tuple_delimiter}"Geometric Brownian Motion"{tuple_delimiter}"Model"{tuple_delimiter}"A mathematical model used to describe the stochastic process of an exchange rate in the Foreign Exchange market."){record_delimiter}
("entity"{tuple_delimiter}"dS_t = (r_d - r_f)S_t dt + σS_t dW_t"{tuple_delimiter}"Formula"{tuple_delimiter}"The stochastic differential equation (SDE) that mathematically defines the evolution of the exchange rate S_t under the Geometric Brownian Motion model."){record_delimiter}
("entity"{tuple_delimiter}"r_d"{tuple_delimiter}"Variable"{tuple_delimiter}"A variable representing the domestic interest rate."){record_delimiter}
("entity"{tuple_delimiter}"r_f"{tuple_delimiter}"Variable"{tuple_delimiter}"A variable representing the foreign interest rate."){record_delimiter}
("entity"{tuple_delimiter}"σ"{tuple_delimiter}"Variable"{tuple_delimiter}"A variable representing the volatility of the exchange rate."){record_delimiter}
("relationship"{tuple_delimiter}"Geometric Brownian Motion"{tuple_delimiter}"dS_t = (r_d - r_f)S_t dt + σS_t dW_t"{tuple_delimiter}"The Geometric Brownian Motion model is mathematically expressed by this specific stochastic differential equation."{tuple_delimiter}"Defines"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Arbitrage Pricing Theory"{tuple_delimiter}"Geometric Brownian Motion"{tuple_delimiter}"The choice of the Geometric Brownian Motion model is part of the standard procedure under Arbitrage Pricing Theory for this market."{tuple_delimiter}"Based_on_assumption"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"arbitrage pricing, geometric brownian motion, stochastic differential equation, foreign exchange, asset pricing"){completion_delimiter}

#############################""",
    r"""Example 2:

Entity_types: [Formula, Variable, Model, Financial_Product, Financial_Concept]
Text:
```
Like in Option pricing: general principles, one can compute closed form solutions for many options types with payoff $F ( S_{{T}} )$ at maturity $T$ directly via

$$
\begin{{aligned}}
v(t, x) &= e^{{-r_d T}} \mathbb{{E}} \left[ F(S_T) \mid S_t = x \right] \\
        &= e^{{-r_d T}} \mathbb{{E}} \left[ F \left( x e^{{(r_d - r_f - \frac{{1}}{{2}} \sigma^2)\tau + \sigma \sqrt{{\tau}} Z}} \right) \right],
\end{{aligned}}
$$

where $v ( t , x )$ denotes the value of the derivative with payoff $F$ at time $t$ if the spot is at $x$ . The random variable $Z$ represents the continuous returns, which are modeled as standard normal in the Black-Scholes model. In this model we can proceed as

$$
\begin{{array}}{{lll}}
v ( t , x ) & = & e^{{ - r_{{d}} \tau }} \displaystyle \int_{{ - \infty }}^{{ + \infty }} F \left( x e ^ {{ ( r_{{d}} - r_{{f}} - \frac{{1}}{{2}} \sigma^{{2}} ) \tau + \sigma \sqrt{{\tau}} z }} \right) n ( z ) d z \\
& = & D_{{d}} \displaystyle \int_{{ - \infty }}^{{ + \infty }} F \left( x e ^ {{ \sigma \theta_{{ - }} \tau + \sigma \sqrt{{\tau}} z }} \right) n ( z ) d z .
\end{{array}}
$$
```

Output:
("entity"{tuple_delimiter}"v(t, x) = e^(-r_d T) * E[F(S_T) | S_t = x]"{tuple_delimiter}"Formula"{tuple_delimiter}"A general formula for the value of a derivative, defined as the discounted expectation of its future payoff F(S_T) at maturity T."){record_delimiter}
("entity"{tuple_delimiter}"Black-Scholes Model"{tuple_delimiter}"Model"{tuple_delimiter}"A financial model where continuous returns (represented by Z) are assumed to be standard normal, enabling the calculation of derivative values via integration."){record_delimiter}
("entity"{tuple_delimiter}"Integral Valuation Formula"{tuple_delimiter}"Formula"{tuple_delimiter}"v(t,x) = e^(-r_d*tau) * integral(F(...)*n(z)dz). This formula is the specific implementation of the general valuation principle within the Black-Scholes model."){record_delimiter}
("entity"{tuple_delimiter}"Z"{tuple_delimiter}"Variable"{tuple_delimiter}"A random variable that represents the continuous returns of the underlying asset."){record_delimiter}
("relationship"{tuple_delimiter}"v(t, x) = e^(-r_d T) * E[F(S_T) | S_t = x]"{tuple_delimiter}"Integral Valuation Formula"{tuple_delimiter}"The Integral Valuation Formula is the result of solving the expectation in the general formula under the assumptions of the Black-Scholes model."{tuple_delimiter}"Derives_from"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Black-Scholes Model"{tuple_delimiter}"Z"{tuple_delimiter}"The Black-Scholes model is based on the assumption that the random variable Z, representing asset returns, follows a standard normal distribution."{tuple_delimiter}"Based_on_assumption"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"option pricing, closed-form solution, Black-Scholes, risk-neutral valuation, integral formula"){completion_delimiter}

#############################""",
]

PROMPTS["summarize_entity_descriptions"] = """
你的角色是一名知识整合专家。你的任务是根据下方提供的实体描述列表，提炼并综合出一个全面、连贯且信息丰富的实体（或多个实体）简介。

---核心要求---
1.  **信息融合**：你需要有机地融合所有描述中的独特信息，构建出实体的完整画像。避免简单拼接描述内容，而是要将各个要点逻辑地串联起来。
2.  **冲突处理**：如果描述之间存在矛盾，首先判断它们是否代表了实体的不同方面，或是某一广义概念的具体情形（如一般定义与特定模型下的应用）。基于此分析，给出最全面、最合理的统一解释。
3.  **叙述风格**：最终输出应为一段客观、中立、第三人称视角的单段文字。请确保在描述中包含实体名称，以保证上下文完整。
4.  **输出语言**：请使用{language}输出结果。

#######
---数据---
实体名称: {entity_name}
描述列表: {description_list}
#######
输出:
"""

PROMPTS["entity_continue_extraction"] = """
上一次抽取遗漏了许多实体和关系。请仅从前面的文本中找出遗漏的实体和关系。

---请遵循以下步骤---

1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name：实体名称，保持与输入文本相同的语言。如果为英文，首字母大写
- entity_type：实体类型，必须为以下类型之一：[{entity_types}]
- entity_description：仅基于输入文本中明确给出的信息，提供该实体的全面描述。**不要推断或臆造文本中未明确说明的信息。**如果文本信息不足以给出完整描述，请写“文本中未提供描述”。
每个实体的格式如下：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 在第1步识别的实体中，找出所有*明确相关*的实体对（source_entity, target_entity）。
对于每一对相关实体，提取以下信息：
- source_entity：源实体名称，来自第1步
- target_entity：目标实体名称，来自第1步
- relationship_description：说明你认为这两个实体相关联的原因
- relationship_strength：用数字表示这两个实体之间关系的强度
- relationship_keywords：用一个或多个高层次关键词总结该关系的本质，聚焦于概念或主题而非具体细节
每个关系的格式如下：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 总结文本的主要概念、主题或话题，提取高层次关键词，反映文档的核心思想。
格式如下：("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 按{language}输出第1步和第2步中所有实体和关系，作为一个列表。用**{record_delimiter}**作为列表分隔符。

5. 完成后输出{completion_delimiter}

---输出格式---

请仅在下方补充新识别出的实体和关系，不要重复已抽取过的内容，格式保持一致：\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---目标---

系统检测到可能仍有实体未被抽取完全。

---输出要求---

请仅回答“YES”或“NO”，判断是否还有需要补充的实体。
""".strip()

PROMPTS["fail_response"] = (
    "很抱歉，我无法针对该问题提供答案。[no-context]"
)

PROMPTS["rag_response"] = """---角色---

你是一名专注于量化金融与数学领域的专家助手。你的任务是利用JSON格式提供的知识图谱和文档片段，回答用户的问题。

---目标---
基于所提供的知识库，生成简明且准确的回答，并遵循以下所有规则。你的回答应综合知识库中的信息，同时可结合你自身的通用知识，特别是在数学公式的表达上优化内容呈现。

---意图识别机制---
1. 首先分析用户问题的核心意图和信息需求
2. 检查context_data中是否包含与问题直接相关的信息
3. 如果context_data中没有相关信息，可以使用你自身训练数据的知识库进行回答
4. 但在这种情况下，必须明确说明：“在所提供的知识库中未找到相关信息，以下回答基于我的训练数据知识库。”

---公式处理规则---
- 必须认真检查知识库中检索到的所有LaTeX公式，这些公式可能存在格式上的小瑕疵。
- 利用你对数学和科学记号的通用知识，对这些公式进行修正和美化，使其表达更清晰、专业。
- 例如，如果上下文暗示统计期望，应将简单的`E[...]`规范为标准LaTeX `\mathbb{{E}}[...]`。同样，确保变量、向量、运算符和函数（如`sin`、`log`、`exp`）均采用标准且高质量的LaTeX格式。
- 这种优化仅限于*公式的表达方式*，不改变其数学含义或逻辑。

---对话历史---
{history}

---知识图谱与文档片段---
{context_data}

---回答规则---
- 目标格式与长度：{response_type}
- 使用Markdown格式，合理分节，并确保所有LaTeX公式正确包裹（如$...$或$$...$$）。
- 请使用与用户问题相同的语言进行回答。
- 回答需与对话历史保持连贯。
- 当引用context_data中的信息时，在回答结尾以“参考文献”部分列出最多5个最重要的参考来源，仅列出文档片段（DC）来源，格式如下：[DC] 文件路径（章节信息）。示例：[DC] financial_report.pdf（第2.3节 市场分析）
- 如果context_data中没有解答所需的信息，必须使用你自身训练数据的知识库进行回答，并在回答开头明确说明：“在所提供的知识库中未找到相关信息，以下回答基于我的训练数据知识库。”
- **特别注意：当使用你自身知识库时，严禁编造新的事实信息。** 只能依赖训练数据中的可靠知识。
- **但对已有信息的格式修正，尤其是数学公式的美化，属于例外且必须执行。**
- 用户补充提示：{user_prompt}

回答："""

PROMPTS["keywords_extraction"] = """---角色---

你是一名智能助手，负责从用户的提问和对话历史中提取高层次和低层次关键词。

---目标---

根据用户的提问和对话历史，分别列出高层次关键词和低层次关键词。高层次关键词关注整体概念或主题，低层次关键词关注具体实体、细节或专有名词。

---操作说明---

- 提取关键词时需同时考虑当前提问和相关的对话历史
- 输出必须为JSON格式，系统会用JSON解析器解析你的输出，禁止输出任何额外内容
- JSON需包含两个键：
  - "high_level_keywords"：用于整体概念或主题
  - "low_level_keywords"：用于具体实体或细节

######################
---示例---
######################
{examples}

######################
---真实数据---
######################
对话历史:
{history}

当前提问: {query}
######################
输出内容必须为JSON格式，且前后不得有任何其他文本。请使用与“当前提问”相同的语言输出。

输出:
"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

PROMPTS["naive_rag_response"] = """---角色---

你是一名智能助手，负责根据下方提供的文档片段（Document Chunks，已以JSON格式给出）回答用户问题。

---目标---

请基于文档片段内容，结合对话历史和当前提问，生成简明扼要的回复，并遵循“回复规则”。需全面总结所有文档片段中的信息，并可适当结合与文档片段相关的常识，但不得引入文档片段未提供的信息。

处理带有时间戳的内容时：
1. 每条内容都包含“created_at”时间戳，表示我们获取该知识的时间
2. 如遇内容冲突，请同时考虑内容本身和时间戳
3. 不要机械地优先采用最新内容，应结合上下文进行判断
4. 针对时间敏感的问题，优先参考内容中的时间信息，其次才考虑获取时间

---对话历史---
{history}

---文档片段（DC）---
{content_data}

---回复规则---

- 回复格式与长度要求：{response_type}
- 使用Markdown格式，并添加合适的小节标题
- 回复请使用与用户提问相同的语言
- 保证回复内容与对话历史连贯
- 在回复末尾以“参考文献”小节列出最多5条最重要的参考来源，需明确标注每条来源自文档片段（DC），并尽量包含文件路径，格式如下：[DC] file_path
- 如果你不知道答案，请直接说明
- 不得包含文档片段未提供的信息
- 额外用户提示：{user_prompt}

回复："""

# TODO: deprecated
PROMPTS["similarity_check"] = """请分析以下两个问题的相似度：

问题1：{original_prompt}
问题2：{cached_prompt}

请判断这两个问题在语义上是否相似，以及问题2的答案是否可以用于回答问题1，并直接给出一个0到1之间的相似度分数。

相似度评分标准：
0：完全无关或答案不可复用，包括但不限于：
   - 问题主题不同
   - 涉及的地点不同
   - 涉及的时间不同
   - 涉及的具体人物不同
   - 涉及的具体事件不同
   - 问题背景信息不同
   - 关键条件不同
1：完全相同，答案可直接复用
0.5：部分相关，答案需修改后才能使用

只返回一个0-1之间的数字，不要输出任何额外内容。
"""
