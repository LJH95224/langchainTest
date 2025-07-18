from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(verbose= True, override=True)
from langchain_openai import ChatOpenAI

siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

# client = OpenAI(api_key=siliconflow_api_key, base_url=siliconflow_base_url)
# response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-72B-Instruct",
#     messages=[
#         {'role': 'user',
#         'content': "推理模型会给市场带来哪些新的机会"}
#     ],
#     stream=True
# )


llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0,
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

# # invoke
# response = llm.invoke("推理模型会给市场带来哪些新的机会")
# print(response)


for chunk in llm.stream("推理模型会给市场带来哪些新的机会"):
    print(chunk)