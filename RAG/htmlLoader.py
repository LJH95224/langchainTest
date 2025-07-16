import bs4
from langchain_community.document_loaders import WebBaseLoader
import asyncio


async def main():
    page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
    loader = WebBaseLoader(web_paths=[page_url])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    assert len(docs) == 1
    doc = docs[0]

    print(f"{doc.metadata}\n")
    print(doc.page_content[:500].strip())


asyncio.run(main())