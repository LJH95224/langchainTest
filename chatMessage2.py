from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pprint import pprint
# 直接创建消息

# 系统消息，代表设定 AI 的角色和行为，比如“你是一个帮助用户起名的大师”。
sy = SystemMessage(
    content="你是一个起名大师",
    additional_kwargs={"大师姓名": "陈瞎子"}
)

# 用户（人类）发送的消息，通常是提问或指令。
humm = HumanMessage(
    content="请问大师叫什么"
)

# AI 给出的回复。
ai = AIMessage(
    content="我叫陈瞎子"
)
pprint([sy, humm, ai])


"""
[
SystemMessage(content='你是一个起名大师', additional_kwargs={'大师姓名': '陈瞎子'}, response_metadata={}),
HumanMessage(content='请问大师叫什么', additional_kwargs={}, response_metadata={}),
AIMessage(content='我叫陈瞎子', additional_kwargs={}, response_metadata={})]
"""