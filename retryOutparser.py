from langchain.output_parsers import RetryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv(verbose=True, override=True)

# 初始化模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# 定义 Pydantic 输出结构
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# 创建输出解析器
parser = PydanticOutputParser(pydantic_object=Action)

# 创建 PromptTemplate
prompt = PromptTemplate(
    template="""Based on the user question, provide an Action and Action Input for what step should be taken.

{format_instructions}

Question: {query}
Response:""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 构造提示词内容
prompt_value = prompt.format_prompt(query="北京今天天气如何？")

# 假设错误返回：字段缺失（action_input缺失）
bad_response = '{"action": "search"}'

# 手动触发解析失败
print("\n--- 尝试解析 bad_response ---")
try:
    result = parser.parse(bad_response)
    print("✅ 成功解析：", result)
except OutputParserException as e:
    print("❌ 解析失败：", e)

# 使用 RetryOutputParser 自动重试
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)

# 自动修正错误响应（会调用 llm 补全）
print("\n--- RetryOutputParser 自动重试 ---")
try:
    result = retry_parser.parse_with_prompt(bad_response, prompt_value)
    print("✅ 最终解析结果：", result)
except OutputParserException as e:
    print("❌ 即使重试也解析失败：", e)