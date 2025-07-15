import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
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
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)
# 事件列表
# events = []
#
# async def main():
#     async for event in llm.astream_events("LanginChain 的作者是谁？"):
#         events.append(event)
#         print(f"event: {event['event']} | name={event['name']} | data={event['data']}")
#
#     print(events)
#
# asyncio.run(main())


# 事件过滤 按 name
"""
•    先用ChatDeepSeek模型生成结果（run_name为model）。
•    然后用JsonOutputParser解析返回结果（run_name为my_parser）。
•    你监听的是"my_parser"阶段的事件。
"""
# chain = llm.with_config({"run_name": "model"}) | JsonOutputParser().with_config({"run_name": "my_parser"})
#
# async def main():
#     max_events = 0
#     async for event in chain.astream_events("LangChain 的作者是谁？", include_names=["my_parser"], version="v2"):
#         print(f"event: {event['event']} | name={event['name']} | data={event['data']}")
#         max_events += 1
#         if max_events > 10:
#             print("...........")
#             break
#
# asyncio.run(main())


# 事件过滤 按 tag
#
# chain = chain = (
#     llm.with_config({"tags": ["llm", "my_chain"]})
#     | JsonOutputParser().with_config({"tags": ["parser", "my_chain"]})
# )
#
# async def main():
#     max_events = 0
#     async for event in chain.astream_events("LangChain 的作者是谁？", include_tags=["my_chain"], version="v2"):
#         print(f"\n[Event] {event['event']}")
#         print(f"Name: {event.get('name')}")
#         print(f"Tags: {event.get('tags')}")
#         print("Data:", event.get("data"))
#         max_events += 1
#         if max_events > 10:
#             print("...........")
#             break
#
# asyncio.run(main())


# 事件阶段过滤

chain = llm | JsonOutputParser()

async def main():
    max_events = 0
    async for event in chain.astream_events("LangChain 的作者是谁？", version="v2"):
        print(f"\n[Event] {event['event']}")
        print(f"Name: {event.get('name')}")
        print(f"Tags: {event.get('tags')}")
        print("Data:", event.get("data"))

        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(f"\n[Event] {event['event']}")
            print(f"Name: {event.get('name')}")
            print(f"Tags: {event.get('tags')}")
            print("Data:", event.get("data"))
        if kind == "on_parser_stream":
            print(f"\n[Event] {event['event']}")
            print(f"Name: {event.get('name')}")
            print(f"Tags: {event.get('tags')}")
            print("Data:", event.get("data"))

        max_events += 1
        if max_events > 30:
            print("...........")
            break

asyncio.run(main())