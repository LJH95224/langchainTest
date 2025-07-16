# 多模态PDF
# 对于包含图片内容的PDF可以使用多模态或者 unstructured 解析

import base64
import io
import sys
import fitz
import os
from  PIL import Image
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(verbose=True, override=True)

def pdf_page_to_base64(pdf_path: str, page_number: int):
    """
    函数 pdf_page_to_base64(pdf_path: str, page_number: int) 实现了：
	•	用 PyMuPDF（即 fitz）打开 PDF
	•	渲染指定页为图像（像素图）
	•	使用 PIL 转成 PNG
	•	编码为 base64 字符串，便于前端传输或喂给大模型
    """
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def save_base64_to_image(base64_str, output_path):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)

    # 自动打开图片（适配不同操作系统）
    if sys.platform.startswith('darwin'):  # macOS
        os.system(f'open {output_path}')
    elif sys.platform.startswith('linux'):
        os.system(f'xdg-open {output_path}')
    elif sys.platform.startswith('win'):
        os.startfile(output_path)

file_path = '../resource/DeepSeek从入门到精通(20250204).pdf'

base64_image = pdf_page_to_base64(file_path, 3)
# print(IPImage(base64.b64decode(base64_image)))

# 保存并自动打开图片
# save_base64_to_image(base64_image, "output_page3.png")




query="这张图片里有什么?"

siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

from openai import OpenAI


client = OpenAI(api_key=siliconflow_api_key, base_url=siliconflow_base_url)

##-------------------------------处理方法1--------------------------------------
# response = client.chat.completions.create(
#     model="deepseek-ai/deepseek-vl2",
#     messages=[
#         {
#             "role": "user",
#              "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                      {
#                          "type": "text",
#                          "text": query
#                      }
#                 ],
#         }
#     ],
#     temperature=0.7,
#     max_tokens=1024,
#     stream=False
# )
# print(response)

##-------------------------------处理方法2--------------------------------------
# response = client.chat.completions.create(
#     model="deepseek-ai/deepseek-vl2",
#     messages=[
#         {
#             "role": "user",
#              "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                      {
#                          "type": "text",
#                          "text": query
#                      }
#                 ],
#         }
#     ],
#     temperature=0.7,
#     max_tokens=1024,
#     stream=True
# )
# # # 逐步接收并处理响应
# for chunk in response:
#     if not chunk.choices:
#         continue
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     if chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end="", flush=True)


##-------------------------------处理方法2--------------------------------------
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出）
    temperature=0,
    model="deepseek-ai/deepseek-vl2",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]
)

response = llm.invoke([message])
print(response)
