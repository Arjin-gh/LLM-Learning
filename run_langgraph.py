"""
@file   : 001.py
@time   : 2026-01-25
"""
# 两阶段的大模型调用
# 一阶段: 复用一阶段的逻辑 识别出菜品 以及 时令
# 二阶段: (1)分词完全匹配过滤黑名单  (2) is_seasonal和timing是否冲突  前面"是" 后面写全年 则过滤。 (3) 基于前面没被过滤的数据做食材的二次判断
# 最后交付阶段的清洗:
# (1): 清洗城市:
#       step1: 匹配全国城市、
#       step2: 判断是否是国外，是国外 改全国。 不是国外 走第三步。
#       step3: 将当前地点写成prompt喂给大模型 输出就近的某几个城市。
import json
import jieba
import pandas as pd
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


def prompt_process_v1(query):
    prompt = f"""
    # Role
你是一位资深的中国餐饮供应链与时令食材专家。

# Task
用户将输入一个中文菜品名称（SPU）<query>
{query}
</query>。你需要分析该菜品及其核心食材，判断其时令性、产地和具体上市时间。

# Logic
1.  **拆解**：分析菜名。如果菜名包含具有明显时令特征的食材（如“冬笋腊肉”包含“冬笋”和“腊肉”），请拆解为多行输出。如果菜名本身就是强时令代表且不可拆分（如“青团”），则作为“菜品”输出。
2.  **判断**：针对每一行（无论是菜品还是食材），进行以下维度的判断：
    * **是否时令**：是/不是。
    * **信息类型**：菜品/食材。
    * **地域**：如果是特产，输出具体城市或省份（如“杭州”、“四川”）；如果是通用食材，输出“全国”。
    * **时令时间**：输出具体的季节+月份范围，或节气，或特定节日。

# required
1. 时令的定义是：食材在特定时间出现，有尝鲜季，比如“春笋”、“冬笋”；或者在某一段时间内受欢迎的菜品，比如清明节时候的“青团”。所以当一个食材所属的"timing"时令时间是全年，则"is_seasonal"为"否"。

# Output Format
请仅输出一个标准的 JSON 格式数组，不要包含任何 Markdown 标记或额外的解释性文字。
JSON 结构如下：
[

    "is_seasonal": "是", // 或 "不是"
    "info_name": "冬笋",
    "info_type": "食材", // 或 "菜品"
    "region": "江浙沪/全国",
    "timing": "冬季（11月至次年2月）"

]

# Example Input
冬笋腊肉

# Example Output
[
  "is_seasonal": "是", "info_name": "冬笋", "info_type": "食材", "region": "江浙、福建", "timing": "冬季（10月下旬至次年2月初）",
  "is_seasonal": "是", "info_name": "腊肉", "info_type": "食材", "region": "四川、湖南、广东", "timing": "冬季/春节（腊月至正月）" 
]
"""
    return prompt


def prompt_process_v2(ingredient):
    prompt = f"""
    你是专业的食材时令性判断助手，需严格按照以下核心逻辑与规则，对用户输入的单一食材名称（含常见别名）进行时令性判定，并输出结构化结果。

### 核心判断三要素（优先级从高到低）
1. **人工供应能力**：能否通过大棚种植、规模化养殖、标准化加工、低温储存等方式，实现全年优质稳定供应，无明显断供期；需特别注意：合法食用蛙类（如克猫儿对应的人工养殖品种）已可通过温控、标准化养殖技术实现全年稳定供应，不受自然水温、饵料等条件限制。
2. **自然周期优势**：自然生长/成熟的食材，其核心风味、口感是否只能在特定季节形成，且人工培育/养殖无法复刻该品质；若人工养殖可通过技术手段（如温控、营养调控）复刻自然品质，则不具备不可替代的自然周期优势。
3. **大众季节认知**：消费者是否普遍存在“某季节必吃该食材”的**全国性强共识**，非应季时新鲜优质食材极少流通；仅局部地区的食用传统不视为强共识。

### 判定规则
- **是时令**：需同时满足「人工无法兜底全年优质供应」+「自然周期优势不可替代」+「大众有强季节认知」；
- **不是时令**：满足「人工可实现全年优质供应」或「自然周期优势可被人工替代」或「大众无强季节认知」任一条件即可判定。

### 特殊情况补充规则
1. **食材别名统一处理**：如“苦茭=苦藠=苦蕌”“子姜=仔姜”“雪里红=雪里蕻”，按同一食材判定；
2. **加工类食材**：重点判断原料是否全年可控+加工后是否脱离季节约束，如腊肉、泡菜、豆瓣酱等，默认归为“不是时令”，除非原料为强时令食材且加工后仍无法全年供应；
3. **统称类食材**：如“时蔬”“小海鲜”“杂菌”，因包含多种单品，判定为“无明确时令属性”；
4. **野生/人工养殖区分**：市面流通的“野味”（如野山羊、野兔）若为人工驯化养殖，按“规模化供应”判定为“不是时令”；仅完全依赖野外自然生长且人工无法培育的品种，才纳入“是时令”评估。

### 用户输入食材名称
<ingredient>
{ingredient}
</ingredient>

### 输出格式要求（严格遵循，不得修改格式，内容需要有用户输入ingredient的个性化思考体现）
1.  判定结论：【是时令/不是时令/无明确时令属性】
2.  核心判断依据：
    （1）人工供应能力：[说明能否全年稳定供应，如大棚种植/规模化养殖/加工储存等方式]
    （2）自然周期优势：[说明人工是否可复刻核心品质，自然季优势是否不可替代]
    （3）大众季节认知：[说明是否有普遍的季节食用共识]
3.  总结：[一句话概括判定逻辑]

请严格按照上述规则和格式输出判断结果，不得添加任何无关内容。
"""
    return prompt


llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key="sss",
    model='deepseek-chat',
    temperature=0.95
)


# spu: 菜品名   spu_parse:     is_seasonal:
class MyState(TypedDict):
    spu: str
    spu_parse: str
    spu_parse_cleaned: list
    result: dict


def run_first_spu_parse(state):
    messages = [
        SystemMessage(content="你是一个餐饮ai助手，可以回答用户的所有问题"),
        HumanMessage(content=prompt_process_v1(state['spu']))
    ]
    result = llm.invoke(messages)
    spu_parse = result.content
    return {"spu_parse": spu_parse}



def load_black_vocab(path):
    df = pd.read_excel(path, header=None)
    black_vocab_list = df[0].tolist()
    result = []
    for v in black_vocab_list:
        temp = jieba.lcut(v)
        temp = set(temp)
        result.append(temp)
    return result
black_vocab_list = load_black_vocab("时令食材黑名单.xlsx")

def is_black_ingredient(ingredient):
    # 判断
    ingredient_seg_list = jieba.lcut(ingredient)
    ingredient_seg_set = set(ingredient_seg_list)

    for black_set in black_vocab_list:
        temp = ingredient_seg_set - black_set
        if len(temp) == 0:
            return True   # 黑名单
    return False  # 不是黑名单


def clean_spu_parse_result(state):
    try:
        spu_parse = json.loads(state['spu_parse'])   # '{"age": 18, "name": "bob"}' => {"age": 18, "name": "bob"}    str=>dict
    except:
        return {"spu_parse_cleaned": []}
    # list
    # black_vocab_list
    result = []
    for item in spu_parse:
        is_seasonal = item.get("is_seasonal")
        timing = item.get("timing")
        info_type = item.get("info_type")
        info_name = item.get("info_name")   # 食材名字

        # 0. 数据不规范的过滤
        if info_name is None or is_seasonal is None or timing is None or info_type is None :
            continue

        # 1. 过滤黑名单
        is_black = is_black_ingredient(info_name)
        if is_black:
            continue

        # 2. 必须要保证是 食材 并且 前后不矛盾
        if (is_seasonal == '是') and ("全年" not in timing) and (info_type == '食材'):
            result.append(info_name)
    return {"spu_parse_cleaned": result}


def run_second_is_seasonal(state):
    '''
    [{'is_seasonal': '是', 'info_name': '东北芹菜', 'info_type': '食材', 'region': '黑龙江、吉林、辽宁', 'timing': '秋季至冬季（9月至次年1月）'},
     {'is_seasonal': '是', 'info_name': '河南木耳', 'info_type': '食材', 'region': '河南', 'timing': '春季至秋季（4月至10月）'}]
    '''
    spu_parse_cleaned = state['spu_parse_cleaned']
    # "你是一个餐饮ai助手，可以回答用户的所有问题"   #
    final_result = {}
    for ingredient in spu_parse_cleaned:
        # 封装prompt
        messages = [
            SystemMessage(content="你是一个餐饮ai助手，可以回答用户的所有问题"),
            HumanMessage(content=prompt_process_v2(ingredient))
        ]
        result = llm.invoke(messages)
        is_seasonal_res = result.content
        final_result[ingredient] = is_seasonal_res
    # {'菜品1': '分析结果', "菜品2": "分析结果", .....}
    return {"result": final_result}


def build_workflow():
    workflow = StateGraph(MyState)

    # 1. 加节点
    workflow.add_node("菜品解析", run_first_spu_parse)
    workflow.add_node("清洗过滤", clean_spu_parse_result)
    workflow.add_node("时令判断", run_second_is_seasonal)

    # 2. 加边
    workflow.add_edge(START, "菜品解析")
    workflow.add_edge("菜品解析", "清洗过滤")
    workflow.add_edge("清洗过滤", "时令判断")
    workflow.add_edge("时令判断", END)
    return workflow.compile()


if __name__ == '__main__':
    workflow = build_workflow()  # 构造工作流
    query = "咸肉春笋烧黄鳝."
    res = workflow.invoke({"spu": query})
    print(res)
