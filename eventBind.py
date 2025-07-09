import os
import asyncio
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

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
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)
question = "你好, 帮我介绍一下你自己？"
question2 = "LanginChain 的作者是谁？"
"""
# invoke 单次调用（同步） 输入list，输出list
response = llm.invoke(question)
print(response)
"""

"""
# stream 流式调用（边生成边接收）
for chunk in llm.stream(question):
    print(chunk.content)
"""

"""
# batch  批量调用（多个输入一起处理）
response = llm.batch([question, question2])
print(response)
"""

# astream_events异步流事件
async def main():
    async for event in llm.astream_events("LanginChain 的作者是谁？", version="v2"):
        print(f"event: {event['event']} | name={event['name']} | data={event['data']}")

asyncio.run(main())

