import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

load_dotenv(verbose=True, override=True)

llm = ChatDeepSeek(
    model="deepseek-chat",
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出【更有创意】）0.7为一个阈值
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

load_dotenv(verbose=True, override=True)
siliconflow_api_key = os.environ["SILLICONFLOW_API_KEY"]
siliconflow_base_url = os.environ["SILLICONFLOW_API_BASE"]

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-4B",
    openai_api_key=siliconflow_api_key,
    openai_api_base=siliconflow_base_url,
)

def pretty_print_docs(docs):
    print(
        f"\n {'-' * 10} \n".join(
            [f"Document{i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# ----------------------------------------未优化前使用向量数据库自带能力------------------------------------------------
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import CharacterTextSplitter

text_path = "../resource/Agent.txt"
documents = TextLoader(text_path).load()
text_splitter = CharacterTextSplitter(
    chunk_size=300,  # 小于限制
    chunk_overlap=50  # 保留部分上下文
)
texts = text_splitter.split_documents(documents)
# 创建向量数据库使用自带的检索器
retriever = FAISS.from_documents(texts, embeddings).as_retriever()

# -----------------------------------使用基础检索器-----------------------------------------------------------
# 使用基础检索器
# docs = retriever.invoke("简单介绍一下 多Agent?")
# pretty_print_docs(docs)

# -----------------------------------使用 LLMChainExtractor-----------------------------------------------------------

# 使用 LLMChainExtractor
# 基础检索器 ContextualCompressionRetriever 以及 LLMChainExtractor 它将迭代最初返回的文档，并从每个文档中提取与查询相关的内容

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
#
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
#
# # 使用 ContextualCompressionRetriever
# compressed_doc = compression_retriever.invoke("简单介绍一下 多Agent?")
#
# pretty_print_docs(compressed_doc)

# -----------------------------------LLMChainFilter-----------------------------------------------------------
# LLMChainFilter: 使用LLM链来决定过滤掉哪些最初检索到的文档以及返回哪些文档，而无需操作文档内容

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainFilter
#
# _filter = LLMChainFilter.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)
#
# compressed_doc = compression_retriever.invoke("简单介绍一下 多Agent?")
# pretty_print_docs(compressed_doc)

#-------------------------------------多个压缩器组合管道----------------------------------------------------------
# 多个压缩器组合管道
# - 使用DocumentCompressorPipeline 轻松地按顺序组合多个压缩器
# - 将文档拆解成更小的碎片
# - 删除冗余文档
# - 串联多个压缩器

from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("简单介绍一下 多Agent?")
pretty_print_docs(compressed_docs)