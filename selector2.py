# 使用最大余弦相似度来检索相关示例，以使示例尽量符合输入
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


load_dotenv(verbose=True, override=True)

llm = ChatDeepSeek(
    model="deepseek-chat",
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出【更有创意】）0.7为一个阈值
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)

# 构造提示词模版
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n 反义：{output}",
)

# 假设已经有很多的提示词示例组 包含各种性质的反义词
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 传入示例组
    examples=examples,
)