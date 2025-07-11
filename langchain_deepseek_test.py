import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)
# deepseek-r1 模型
# llm = ChatDeepSeek(
#     model="deepseek-reasoner",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     # 最大重试次数
#     max_retries=2,
#     api_key=os.getenv("DEEP_SEEK_API_KEY"),
#     api_base=os.getenv("DEEP_SEEK_API_BASE"),
# )

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
response = llm.invoke("你好, 帮我介绍一下你自己")
print(response)
