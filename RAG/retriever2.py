# 检索器，基本检索器设置
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv(verbose=True, override=True)
siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

embeddings = OpenAIEmbeddings(
    model="netease-youdao/bce-embedding-base_v1",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
    chunk_size=32  # ✅ 限制批量嵌入大小，避免超限
)

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# load blog
loader = WebBaseLoader("https://python.langchain.com/docs/how_to/MultiQueryRetriever/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)

question = "如何让用户查询更准确？"

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm
)

import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


unique_docs = retriever_from_llm.invoke(question)
print(f"unique_docs :{unique_docs}\n")
print(len(unique_docs))