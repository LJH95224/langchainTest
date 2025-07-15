# 在链中使用lambda函数
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
# 导入 chain 装饰器， 用于创建自定义链
from langchain_core.runnables import RunnableLambda
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

def length_function(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


# 创建一个简单的聊天提示模版
prompt = ChatPromptTemplate.from_template(
    "什么是 {a} + {b}"
)


# 处理 'a' 参数
# 1. 从输入字典中提取 foo 键的值
# 1.从输入字典中提取"foo"键的值
# 2.将提取的值传递给length_functiol函数(假设这个函数计算字符串长度)":

# 处理"b"参数:
# 1.创建一个包含两个键值对的字典:
#  - "text1": 从输入字典中提取"foo"键的值
#  - "text2": 从输入字典中提取"bar"键的值
# 2.将这个字典传递给 multiple_lengtlfunction 函数
#   (假设这个函数计算两个文本的总长度)
chain1 = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {
            "text1": itemgetter("foo"),
            "text2": itemgetter("bar")
        } | RunnableLambda(multiple_length_function)
    } | prompt | llm
)

# 调用链处理流程， 输入一个包含 "foo" 和 "bar" 两个键的输入字典
#整个过程:
# 1.计算"bar"字符串的长度作为a的值
# 2.计算"bar"和"gah"字符串的总长度作为b的值
# 3。将这些值填入提示"what is {a}+ {b}"
# 4.让DeepSeek模型回答这个问题

response = chain1.invoke({"foo": "bar", "bar": "gah"})
print(response)