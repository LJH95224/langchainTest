import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)
# deepseek-r1 模型
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# deepseek-chat 模型
llm2 = ChatDeepSeek(
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

joke_chain = ChatPromptTemplate.from_template(
    "请生成一个关于{topic}的笑话。"
) | llm

poem_chain = ChatPromptTemplate.from_template(
    "请生成一个关于{topic}的诗。"
) | llm2

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

# print(map_chain.get_graph())
# 输出 链式调用的图
map_chain.get_graph().print_ascii()


# 查看提示词
response_prompts = map_chain.get_prompts()
print(response_prompts)

# response = map_chain.invoke({"topic": "兔子"})
# print(response)
