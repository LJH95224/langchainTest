import asyncio

from langchain_community.document_loaders import PyPDFLoader

file_path = "../resource/《Deepseek R1 本地部署完全手册》.pdf"
loader = PyPDFLoader(file_path)
pages = []

def main():
    for page in loader.lazy_load():
         pages.append(page)
         print('--------------page.metadata start------------------')
         print(f"{page.metadata} \n")
         print('--------------page.metadata end------------------')
         print('--------------page.page_content start------------------')
         print(page.page_content)
         print('--------------page.page_content end------------------')

main()