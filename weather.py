import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from pprint import pprint

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

@tool
def get_weather(location: str) -> str:
    """根据 location 地名返回当地实时天气。"""
    print(f"[TOOL CALLED] get_weather({location=})")
    return f"{location} 天气晴朗，气温 22 度"


llm_with_tools = llm.bind_tools([get_weather])

from langchain_core.output_parsers import StrOutputParser
chain = llm_with_tools | StrOutputParser()

response = chain.invoke("北京市今天的天气如何？")
pprint(response)

response = llm_with_tools.invoke("北京市今天的天气如何？")
pprint("==Raw response==")
pprint(response)
