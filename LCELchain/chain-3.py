# 在链中自定义支持流输出的函数

import os
from operator import itemgetter
from turtle import Tbuffer

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
# 导入 chain 装饰器， 用于创建自定义链
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from typing import Iterator, List

from sympy.physics.units import years

load_dotenv(verbose=True, override=True)
# deepseek-chat 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出【更有创意】）0.7为一个阈值
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# 创建一个聊天提示模板，要求生成5个与给定动物相似的动物名称，以逗号分隔
prompt = ChatPromptTemplate.from_template(
    "请生成5个与{animal}相似的动物名称，以逗号分隔，不要包含数组"
)

# 创建一个处理链:提示模板 -> 模型 ->字符串输出解析器
str_chain = prompt | llm | StrOutputParser()

#流式输出结果，输入为"熊"
for chunk in str_chain.stream("熊"):
    print(chunk, end="", flush=True)


# 自定义解析器,将LLM输出的标记迭代器
# 按逗号转换为字符串列表
def split_info_list(input: Iterator[str]) -> Iterator[List[str]]:
    # 保存部分输入，直到遇到逗号
    buffer = ""
    for chunk in input:
        # 将当前块添加到缓冲区
        buffer += chunk
        # 当缓冲区中有逗号的时候
        while "," in buffer:
            # 在逗号处分隔缓冲区
            comma_index = buffer.index(",")
            # 输出逗号之前的所有内容
            yield [buffer[:comma_index].strip()]
            # 保存剩余部分用于下一次迭代
            buffer = buffer[buffer.index(",") + 1:]
        #
        # # 输出最后一块
        # if buffer.strip():
        #     yield [buffer.strip()]
    yield [buffer.strip()]

list_chain = str_chain | split_info_list

for chunk in list_chain.stream("熊"):
    print(chunk)