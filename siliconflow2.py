from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(verbose= True, override=True)

siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

client = OpenAI(api_key=siliconflow_api_key, base_url=siliconflow_base_url)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
        {'role': 'user',
        'content': "推理模型会给市场带来哪些新的机会"}
    ],
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)