# 相似性分数
# - 根据相似性打分过滤
# - 为文档添加分数

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(verbose=True, override=True)
siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

# 初始化chroma 客户端
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db" # 可选参数,指定持久化目录
)

import chromadb
persistent_client = chromadb.PersistentClient()
collections = persistent_client.get_or_create_collection("collection_name")
collections.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

texts = ["a", "b", "c"]
vectors = embeddings.embed_documents(texts)
collections.add(ids=["1", "2", "3"], documents=texts, embeddings=vectors)

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embeddings
)

from uuid import uuid4
from langchain_core.documents import Document

documents = [
    Document(
        page_content="今天在做了一个锅巴土豆。",
        metadata={"source": "tweet"},
        id=1
    ),
    Document(
        page_content="最新研究显示，多喝水有助于提高注意力，尤其是在夏季高温下。",
        metadata={"source": "news"},
        id = 2
    ),
    Document(
        page_content="刚下班就看到彩虹，好兆头！#生活记录",
        metadata={"source": "tweet"},
        id = 3
    ),
    Document(
        page_content="本地社区推出“夜间图书馆”计划，鼓励市民夜读。",
        metadata={"source": "news"},
        id = 4
    ),
    Document(
        page_content="被安利了一个无糖冰淇淋，真的好吃又不胖！",
        metadata={"source": "tweet"},
        id = 5
    ),
    Document(
        page_content="受台风影响，明日全市中小学停课。",
        metadata={"source": "news"},
        id = 6
    ),
    Document(
        page_content="家附近新开了一家猫咖，毛孩子们太治愈了。",
        metadata={"source": "tweet"},
        id = 7
    ),
    Document(
        page_content="国家将投入100亿元用于人工智能基础研究。",
        metadata={"source": "news"},
        id = 8
    ),
    Document(
        page_content="刚才健身房里差点举不动，教练说这是突破的信号！",
        metadata={"source": "tweet"},
        id = 9
    ),
    Document(
        page_content="科学家首次在火星样本中检测到有机物质。",
        metadata={"source": "news"},
        id = 10
    )
]

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents, ids=uuids)

# 使用相似性分数检索
results = vector_store.similarity_search(
    "今天吃什么？", k=3, filter={"source": "tweet"}
)

for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# 为文档添加分数
# 通过一个自定义链，可以为原始文档增加相关性评分
from langchain_core.runnables import chain
from typing import List
@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vector_store.similarity_search_with_score(query))
    for doc, score in  zip(docs, scores):
        doc.metadata["score"] = score
    return docs


results = retriever.invoke("今天吃什么？")
print(f"f 回复:{results}")



