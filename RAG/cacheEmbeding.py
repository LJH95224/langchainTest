# 缓存嵌入的向量数据
import os
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
load_dotenv(verbose=True, override=True)


siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

underlying_embeddings = OpenAIEmbeddings(
    model="netease-youdao/bce-embedding-base_v1",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)
# LocalFileStore 把每个文本 → 嵌入向量的对应关系缓存到本地文件系统中。
# 你设定的缓存目录是 /tmp/langchain_cache，里面会以嵌入模型名称和文本哈希作为 key。
store = LocalFileStore("/tmp/langchain_cache")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings = underlying_embeddings,
    document_embedding_cache=store,
    namespace = underlying_embeddings.model,
)

tmp = list(store.yield_keys())
# 输出当前缓存文件夹中存了哪些key（你可用于检查是否命中缓存）。
print(f"tmp: {tmp}\n")

text_path = "../resource/test.txt"

# TextLoader：读取纯文本文件，返回 Document 列表
raw_documents = TextLoader(text_path).load()
# CharacterTextSplitter：将大段文本按字符数切成若干片段，每段最多 1000 字。
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents 是一个包含若干 Document 的 list，每段将成为一个 embedding 单元。
documents = text_splitter.split_documents(raw_documents)
print(f"documents: {documents}\n")

# 创建向量存储
# 把 documents 中的每一段文本转为 embedding，然后存入 FAISS 索引。 使用的是刚刚包好的 cached_embedder，优先读取缓存，节省开销。
db = FAISS.from_documents(documents, cached_embedder)

print(f"db: {db}\n")

# 因为这时候文档已经在缓存中，第二次构建几乎不耗时间、不产生 embedding 费用。
db2 =  FAISS.from_documents(documents, cached_embedder)

print(f"db2: {db2}\n")

# 查看缓存
print(f"tmp: {list(store.yield_keys())[:5]}\n")