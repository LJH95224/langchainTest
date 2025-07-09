# 对话模版具有结构

from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个起名大师， 你的名字叫{name}。"),
        ("human", "你好{name}, 你感觉如何？"),
        ("ai", "你好！我状态非常好！"),
        ("human", "你叫什么名字呢？"),
        ("ai", "你好！我叫{name}"),
        ("human", "{user_input}")
    ]
)

chats = chat_template.format_messages(name="陈大师", user_input="你的爸爸是谁呢")


for msg in chats:
    print(f"{msg.__class__.__name__}: {msg.content}")


pprint(chats)
"""
print(chats) # 打印结果，可能打印结果在终端中格式错乱或被截断了
[
SystemMessage(content='你是一个起名大师， 你的名字叫陈大师。', additional_kwargs={}, response_metadata={}),
HumanMessage(content='你好陈大师, 你感觉如何？', additional_kwargs={}, 
age(content='你好！我状态非常好！', additional_kwargs={}, response_metadata={}),
HumanMessage(content='你叫什么名字呢？', additional_kwargs={}, response_metadata={}),
AIMessage(condditional_kwargs={}, response_metadata={}),
HumanMessage(content='你的爸爸是谁呢', additional_kwargs={}, response_metadata={})]
"""

"""
pprint(chats) # pprint 是 Python 标准库中的一个模块，全称是 “pretty-print”，用于美观地打印复杂的数据结构，比如嵌套的列表、字典、对象等，比默认的 print() 更易读。
[
    SystemMessage(content='你是一个起名大师， 你的名字叫陈大师。', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='你好陈大师, 你感觉如何？', additional_kwargs={}, response_metadata={}),
    AIMessage(content='你好！我状态非常好！', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='你叫什么名字呢？', additional_kwargs={}, response_metadata={}),
    AIMessage(content='你好！我叫陈大师', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='你的爸爸是谁呢', additional_kwargs={}, response_metadata={})
]
"""

"""
# 用如下方式查看结构就不会被格式污染：
for msg in chats:
    print(f"{msg.__class__.__name__}: {msg.content}")
    
SystemMessage: 你是一个起名大师， 你的名字叫陈大师。
HumanMessage: 你好陈大师, 你感觉如何？
AIMessage: 你好！我状态非常好！
HumanMessage: 你叫什么名字呢？
AIMessage: 你好！我叫陈大师
HumanMessage: 你的爸爸是谁呢
"""