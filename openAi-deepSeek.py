import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 自动搜索并加载.env文件
load_dotenv(verbose=True, override=True)

llm = ChatOpenAI(
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出）
    temperature=0,
    model="deepseek-reasoner",  # 或 deepseek-coder，如支持
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
)
response = llm.invoke("你好, 帮我介绍一下你自己")
print(response)
