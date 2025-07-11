# 自定义的提示词模版

# 函数大师：根据函数名称，查找函数代码，并给出中文的代码说明
import os
import inspect
from langchain_core.prompts import StringPromptTemplate
from pprint import pprint
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)
# 定义一个简单的函数作为示例效果
def hello_world(abc):
    print("Hello World")
    return abc

PROMPT = """\
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称，源代码，中文解释。
函数名称： {function_name}
源代码：{sorce_code}
代码解释：
"""

def get_source_code(func):
    """获得源代码"""
    return inspect.getsource(func)

# 自定义模版的class 从 StringPromptTemplate 继承
class CustomPrompt(StringPromptTemplate):

    def format(self, **keywords) -> str:
        # 获得源代码
        source_code = get_source_code(keywords["function_name"])

        # 生成提示词模版
        prompt = PROMPT.format(
            function_name=keywords["function_name"].__name__,
            sorce_code=source_code,
        )

        return prompt


cPrompt = CustomPrompt(input_variables=["function_name"])
prompt = cPrompt.format(function_name=hello_world)

pprint(prompt)

"""
('你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称，源代码，中文解释。\n'
 '函数名称： hello_world\n'
 '源代码：def hello_world(abc):\n'
 '    print("Hello World")\n'
 '    return abc\n'
 '\n'
 '代码解释：\n')
"""

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
response = llm.invoke(prompt)
pprint(response)

# 返回结果
# AIMessage(content='好的，按照您要求的格式输出：\n\n函数名称：hello_world\n\n源代码：\n```python\ndef hello_world(abc):\n    """打印Hello World并返回输入参数\n    \n    Args:\n      Returns:\n        返回与输入参数相同的值\n    """\n    print("Hello World")\n    return abc\n```\n\n代码解释：\n1. 这是一个名为hello_world的Python函数，接受一个参数abc\n2. 函数首先数abc\n4. 这是一个经典的编程入门示例的变体，在基础功能上增加了参数传递和返回\n5. 函数具有文档字符串(docstring)，说明了参数和返回值\n6. 该函数的主要目的是演示基本函数结构，同时保持输age': {'completion_tokens': 173, 'prompt_tokens': 61, 'total_tokens': 234, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 61}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache', 'id': '999eae69-3fef-40b2-9386-51111b680165', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--60fa939b-b889-4154-96fa-1543d4142303-0', usage_metadata={'input_tokens': 61, 'output_tokens': 173, 'total_tokens': 234, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})


"""
AIMessage(content='好的，按照您要求的格式输出：

函数名称：hello_world

源代码：
```python
def hello_world(abc):
    "打印Hello World并返回输入参数
    
    Args:
      Returns:
        返回与输入参数相同的值
    "
    print("Hello World")
    return abc
    
代码解释：
	1.	这是一个名为hello_world的Python函数，接受一个参数abc
	2.	函数首先数abc
	3.	这是一个经典的编程入门示例的变体，在基础功能上增加了参数传递和返回
	4.	函数具有文档字符串(docstring)，说明了参数和返回值
	5.	该函数的主要目的是演示基本函数结构，同时保持输…
"""

