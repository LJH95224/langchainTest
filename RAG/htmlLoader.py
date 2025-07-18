import bs4
from langchain_community.document_loaders import WebBaseLoader
import asyncio

#----------------------------整个页面 start----------------------------------
# async def main():
#     page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
#     loader = WebBaseLoader(web_paths=[page_url], verify_ssl=False)
#     docs = []
#     async for doc in loader.alazy_load():
#         docs.append(doc)
#
#     assert len(docs) == 1
#     doc = docs[0]
#
#     print(f"{doc.metadata}\n")
#     print(doc.page_content[:500].strip())
#
#
# asyncio.run(main())

#----------------------整个页面 end ----------------------------------------
#----------------------页面部分 start ----------------------------------------

# async def main():
#     page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
#     loader = WebBaseLoader(
#         web_paths=[page_url],
#         # 只解析指定的标签
#         bs_kwargs={
#           "parse_only": bs4.SoupStrainer(class_="theme-doc-markdown markdown")
#         },
#         # 获取标签的文本
#         bs_get_text_kwargs={
#             "separator": " | ",
#             "strip": True
#         },
#         verify_ssl=False
#     )
#     docs = []
#     async for doc in loader.alazy_load():
#         docs.append(doc)
#
#     assert len(docs) == 1
#     doc = docs[0]
#     print(f"{doc.metadata}\n")
#     print(doc.page_content[:500])
#
#
# asyncio.run(main())

#----------------------页面部分 end ----------------------------------------


#----------------------网页结构不熟悉 start --------------------------------------

from langchain_unstructured import UnstructuredLoader

async def main():
    page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
    loader = UnstructuredLoader(web_url=page_url)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1

    for doc in docs[:5]:
        print(f"{doc.page_content}\n")

asyncio.run(main())

#----------------------网页结构不熟悉 end ----------------------------------------