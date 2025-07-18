import os
import httpx
from chromadb.errors import RateLimitError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)

openai_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    max_retries=0
)

deepseek_llm = ChatDeepSeek(
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

# 创建带有备用选项的语言模型
# 使用主模型 （OpenAI）失败，将自动尝试使用备用模型（DeepSeek）
llm = openai_llm.with_fallbacks([deepseek_llm])


# with patch("openai.resources.chat.completions.Completions.create", side_effect=error):
#     try:
#         print(llm.invoke("why did the chicken cross the road"))
#     except RateLimitError:
#         print("Hit error")


print(llm.invoke("鸡为什么过马路"))