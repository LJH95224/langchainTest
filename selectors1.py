# 根据输入的提示词长度综合计算最终长度，智能截取或者添加提示词示例
from langchain.smith.evaluation.name_generation import adjectives
from langchain_core.callbacks import Callbacks
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from pprint import pprint

# 假设已经有很多的提示词示例组
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
    {"input": "高兴", "output": "悲伤"},
]

# 构造提示词模版
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n 反义：{output}",
)

# 调用长度示例选择器
example_selector = LengthBasedExampleSelector(
    # 传入提示词示例组
    examples=examples,
    # 传入提示词模版
    example_prompt=example_prompt,
    # 设置格式化后的提示词最大长度
    max_length=25,
    # 内置的get_text_length, 如果默认分词计算方式不满足，可以自己扩展
    # get_text_length: Callbacks[[str], int] = lambda x: len(re.split("\n\ ", x))
)

# 使用小样本提示词模版来实现动态示例的调用
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"],
)

pprint(dynamic_prompt)
pprint('输入短的')
pprint(dynamic_prompt.format(adjective="big"))

pprint('输入长的')
# 如果输入长度很长，则最终输出会根据长度要求减少
long_str = "bid and huge and massive and large and gigantic and tall and much much much much much much bigger then everyone"
pprint(dynamic_prompt.format(adjective=long_str))