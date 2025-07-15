# 使用 @chain 修饰符快速将函数变为链

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
# 导入 字符串输出解析器
from langchain_core.output_parsers import StrOutputParser
# 导入 chain 装饰器， 用于创建自定义链
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate


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

# 创建第一个提示模版： 请求关于特定主题的笑话
prompt1 = ChatPromptTemplate.from_template(
    "请给我一个关于{topic}笑话，请确保 punchline 是一个抖包袱方式回答 setup 问题，例如谐音，会错意等"
)

# 创建第二个提示模版： 询问笑话的主题是什么
prompt2 = ChatPromptTemplate.from_template(
    "这个笑话{joke}的主题是什么"
)

# 使用@chain 装饰器定义一个自定义链
@chain
def custom_chain(text: str):
    # 步骤1 传入 {topic: text} 构建提示词，
    prompt_val1 = prompt1.invoke({"topic": text})
    # 步骤2 调用 DeepSeek 模型生成完整笑话。
    output1 = llm.invoke(prompt_val1)
    # 步骤3 解析输出，提取模型输出的字符串文本
    parsed_output1 = StrOutputParser().invoke(output1)
    # 步骤4 创建第二个链，传入 {joke: parsed_output1}
    chain2 = prompt2 | llm | StrOutputParser()

    return chain2.invoke({"joke": parsed_output1})


# 使用自定义链
response = custom_chain.invoke("小兔子")
print(response)