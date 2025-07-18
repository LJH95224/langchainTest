# 检索器，基本检索器设置
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(verbose=True, override=True)
siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

embeddings = OpenAIEmbeddings(
    model="netease-youdao/bce-embedding-base_v1",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

text_path = "../resource/test.txt"

documents = TextLoader(text_path).load()

# CharacterTextSplitter：将大段文本按字符数切成若干片段，每段最多 1000 字。
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# documents 是一个包含若干 Document 的 list，每段将成为一个 embedding 单元。
texts = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(texts, embeddings)

# 实例话检索器
retriever = vectorstore.as_retriever()

docs = retriever.invoke("余弦相似度是什么?")
print(f"检索结果为：{docs[0].page_content}")