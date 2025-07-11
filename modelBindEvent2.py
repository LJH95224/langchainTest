from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)

# 工具函数绑定
@tool
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘"""
    return a * b

# 初始化 LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# 绑定工具
tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

# 查询
query = "3乘12是多少？"
response = llm_with_tools.invoke(query)
print(response)

# 如果模型请求了工具调用
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']

    # 执行对应的工具
    tool_map = {
        'add': add,
        'multiply': multiply,
    }

    result = tool_map[tool_name].invoke(tool_args)
    print(f"结果是：{result}")
else:
    print(f"模型回复：{response.content}")

