import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(verbose=True, override=True)

llm = ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    openai_api_key=os.environ["DEEP_SEEK_API_KEY"],
    openai_api_base=os.environ["DEEP_SEEK_API_BASE"],
)

prompt = PromptTemplate.from_template(
    "你是一个起名大师，请模仿示例起3个{country}名字，比如男孩经常被叫做{boy}, 女孩经常被叫做{girl}"
)
message = prompt.format(country="中国特色的", boy="狗蛋", girl="翠花")

print(message)
response = llm.invoke(message)
print(response)
