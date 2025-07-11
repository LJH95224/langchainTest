from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import Tool

load_dotenv(verbose=True, override=True)

# 加法模型参数
class AddParams(BaseModel):
    """加法模型参数"""
    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")

# 加法执行函数
def add_fun(a: int, b: int) -> int:
    return a + b

# 创建 Tool 对象（LangChain 能识别的工具）
add_tool = Tool(
    name="add",
    description="两个整数相加",
    args_schema=AddParams,
    func=lambda params: add_fun(**params),
)



# 乘法模型参数
class MultiplyParams(BaseModel):
    """乘法模型参数"""
    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")

# 乘法执行函数
def multiply_fun(a: int, b: int) -> int:
    return a * b

multiply_tool = Tool(
    name="multiply",
    description="两个整数相乘",
    args_schema=MultiplyParams,
    func=lambda params: multiply_fun(**params),
)

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
tools = [add_tool, multiply_tool]
llm_with_tools = llm.bind_tools(tools)

# 查询
query = "3乘12是多少？"
response = llm_with_tools.invoke(query)
print(response)


"""
content=''
additional_kwargs={'tool_calls': [{'id': 'call_0_6bdf6397-9ceb-4b7e-8439-7a5aee42d2d7', 'function': {'arguments': '{"a":3,"b":12}', 'name': 'multiply'}, 'type': 'function', 'index': 0}], 'refusal': None}
response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 243, 'total_tokens': 265, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 128}, 'prompt_cache_hit_tokens': 128, 'prompt_cache_miss_tokens': 115}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache', 'id': '5427579f-238a-4016-af9a-a05e9cbbb6a1', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}
id='run--79e4ca35-00db-4643-a0c8-02f2be0dcf37-0'
tool_calls=[
    {
        'name': 'multiply',
        'args': {'a': 3, 'b': 12},
        'id': 'call_0_6bdf6397-9ceb-4b7e-8439-7a5aee42d2d7',
        'type': 'tool_call'
    }
]
usage_metadata={'input_tokens': 243, 'output_tokens': 22, 'total_tokens': 265, 'input_token_details': {'cache_read': 128}, 'output_token_details': {}}
"""

# 如果模型请求了工具调用
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']

    # 执行对应的工具
    tool_map = {
        'add': add_tool,
        'multiply': multiply_tool,
    }

    result = tool_map[tool_name].func(tool_args)
    print(f"结果是：{result}")
else:
    print(f"模型回复：{response.content}")
