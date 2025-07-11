# 模型事件绑定  DeepSeek 模型并绑定两个工具（add 和 multiply），然后通过自然语言查询来自动触发相应的工具进行调用。
# 只绑定了工具的参数模型（如 add），但没有告诉模型这些模型要执行什么逻辑。
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)

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

class add(BaseModel):
    """两个整数相加"""
    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")

class multiply(BaseModel):
    """将两个整数相乘"""
    a: int = Field(..., description="第一个整数")
    b: int = Field(..., description="第二个整数")

tools = [add, multiply] # 一个模型可以绑定多个工具
llm_with_tools = llm.bind_tools(tools)
query = "3乘12是多少？"
response = llm_with_tools.invoke(query)
print(response)

"""
response 返回的数据结构
content='' 
additional_kwargs={'tool_calls': [{'id': 'call_0_1ae04e39-bc01-4a70-bd90-f37b15c75fec', 'function': {'arguments': '{"a":3,"b":12}', 'name': 'multiply'}, 'type': 'function', 'index': 0}], 'refusal': None}
response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 244, 'total_tokens': 266, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 192}, 'prompt_cache_hit_tokens': 192, 'prompt_cache_miss_tokens': 52}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache', 'id': '633741a3-b53e-4dde-b208-9e1f6b7ba65b', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}
id='run--b889547e-b485-4ce7-bbaa-03db8c0e68e7-0'
tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_0_1ae04e39-bc01-4a70-bd90-f37b15c75fec', 'type': 'tool_call'}]
usage_metadata={'input_tokens': 244, 'output_tokens': 22, 'total_tokens': 266, 'input_token_details': {'cache_read': 192}, 'output_token_details': {}}
"""
