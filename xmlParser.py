import os
from dotenv import load_dotenv
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import List
from pprint import pprint

# 加载 .env 文件中的环境变量
load_dotenv(verbose=True, override=True)

# 初始化模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# -------------------------------
# 1. 定义输出数据结构（Pydantic 模型）
# -------------------------------
class Character(BaseModel):
    name: str = Field(description="角色名称")
    age: str = Field(description="角色年龄")
    description: str = Field(description="角色简要描述")

class ShortFilm(BaseModel):
    title: str = Field(description="短片标题")
    genre: str = Field(description="影片类型")
    duration: str = Field(description="影片时长")
    logline: str = Field(description="一句话剧情简介")
    characters: List[Character] = Field(description="主要角色列表")

# -------------------------------
# 2. 创建 XML 输出解析器
# -------------------------------
parser = XMLOutputParser(pydantic_object=ShortFilm)

# -------------------------------
# 3. 创建 PromptTemplate 提示词模板
# -------------------------------
prompt = PromptTemplate(
    template="""你是一位电影创作者，请基于以下请求生成一部短片的剧情概要。请以 XML 格式输出，结构如下：
{format_instructions}

用户请求：{query}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# -------------------------------
# 4. 构建调用链：提示 -> 模型 -> XML解析器
# -------------------------------

# xml 解析器展示结果，没有xml的格式的标签
chain = prompt | llm  | parser

chain2 = prompt | llm

# -------------------------------
# 5. 执行调用
# -------------------------------
query = "我想要一部关于时间旅行的感人短片，适合家庭观影"
result = chain.invoke({"query": query})
result2 = chain2.invoke({"query": query})

# -------------------------------
# 6. 打印结果（结构化 Pydantic 对象）
# -------------------------------
# 结构化的pydantic对象
pprint(result)

# -------------------------------
# 7. 输出 XML 结构
# ----------------------------
pprint(result2)