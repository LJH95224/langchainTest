import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, model_validator
from pprint import pprint

load_dotenv(verbose=True, override=True)

# 初始化模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# 定义数据模型
class Joke(BaseModel):
    setup: str = Field(description="笑话中铺垫问题，必须以？结尾")
    punchline: str = Field(description="笑话中铺垫问题的部分，通常是一种抖包袱方式回答铺垫问题，例如谐音，会错意等")

    # 验证器， 你可以根据自己的数据情况进行自定义
    # 注意model=before意思是数据被转成pydantic模型的字段之前，对于原始数据进行验证。
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        if setup and not (setup.endswith("?") or setup.endswith("？")):
            raise ValueError("Badly formed question!")
        return values

# 创建解析器与提示词模板
parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="""请讲一个笑话，要求返回 JSON 数据，字段结构如下：
{format_instructions}
用户问题：{query}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 链式组合：提示 -> 模型 -> 输出解析
chain = prompt | llm | parser

# 调用链
# result = chain.invoke({"query": "告诉我一个笑话"})
# pprint(result)

for chunk in chain.stream({"query": "告诉我一个笑话"}):
    pprint(chunk)