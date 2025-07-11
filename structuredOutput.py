# LLM的标准事件 - with_structured_output
# 影响LLM的输出： 以结构化的数据来输出
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from typing import Optional

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

# pydamtic 数据
class Joke(BaseModel):
    """Joke to tell user"""
    setup: str = Field(description="笑话的设定")
    punchline: str = Field(description="笑话的妙语")
    # Optional[int] (可选项，整数型)
    rating: Optional[int] = Field(
        default=None,
        description="这个笑话有多好笑，从1到10评分"
    )


question = "给我将一个关于程序员的笑话"

# invoke
# structured_llm = llm.with_structured_output(Joke)
# response = structured_llm.invoke(question)
# print(response)


# stream 流式
structured_llm = llm.with_structured_output(Joke)
for chunk in structured_llm.stream(question):
    print(chunk)