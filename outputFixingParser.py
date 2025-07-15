from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

load_dotenv(verbose=True, override=True)

# 初始化模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# 定义 Actor 模型
class Actor(BaseModel):
    name: str = Field(description="演员名称")
    film_names: List[str] = Field(description="他们主演的电影名单")

# 错误格式的响应
misFormatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

# 创建 Pydantic parser
parser = PydanticOutputParser(pydantic_object=Actor)

# 测试错误解析
try:
    parser.parse(misFormatted)
    print("✅ 没有抛出错误")
except OutputParserException as e:
    print("❌ 抛出了错误:", e)

# 用 LLM 修复结构问题
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# 自动修复错误格式
result = fixing_parser.parse(misFormatted)
print("✅ 修复后的结果:", result)