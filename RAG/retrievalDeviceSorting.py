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

from langchain_core.vectorstores import InMemoryVectorStore

texts = [
    "西湖是杭州著名的旅游景点。",
    "我最喜欢的歌曲是《月亮代表我的心》。",
    "故宫是北京最著名的古迹之一。",
    "这是一篇关于北京故宫历史的文档。!",
    "我非常喜欢去电影院看电影。",
    "北京故宫的藏品数量超过一百万件。!",
    "这只是一段随机文本。",
    "《三国演义》是中国四大名著之一。",
    "紫禁城是故宫的别称，位于北京。",
    "故宫博物院每年接待游客数百万人次。"
]

# 创建检索器
retriever = InMemoryVectorStore.from_texts(texts, embedding= embeddings).as_retriever(
    search_kywargs={"k": 10}
)

query = "请告诉我关于故宫的消息？"

# 获取相关性排序的文档
docs = retriever.invoke(query)
for doc in docs:
    print(f"📄  {doc.page_content}")


from langchain_community.document_transformers import LongContextReorder

# 重新排序文档：
# 相关性较低的文档将位于列表中间
# 相关性较高的文档将位于开头和结尾

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# 确认相关性高的文档位于开头和结尾
for doc in reordered_docs:
    print(f"-  {doc.page_content}")

# 整合到 chain 里面去

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

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

prompt_template = """
    给定以下文本:
    -----
    {context}
    -----
    请回答以下问题:
    {query}
    """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"]
)

chain = create_stuff_documents_chain(llm, prompt)
response = chain.invoke({"context": reordered_docs, "query": query})
print(f"\n 回复:{response}")