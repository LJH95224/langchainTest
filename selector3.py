# 使用最大余弦相似度来检索相关示例，以使示例尽量符合输入
# 没有openAI 的APi key 运行失败
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
from pprint import pprint

load_dotenv(verbose=True, override=True)

api_key=os.getenv("DEEP_SEEK_API_KEY")
api_base=os.getenv("DEEP_SEEK_API_BASE")

# 构造提示词模版
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="原词：{input}\n 反义：{output}",
)

# 假设已经有很多的提示词示例组 包含各种性质的反义词
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
    {"input": "高兴", "output": "悲伤"},
]

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 传入示例组
    examples=examples,
    # 使用 openAI 嵌入来做相似度搜索
    embeddings = embeddings_model,
    # 使用Chroma向   量数据库来实现对相似结果的过程存储
    vectorstore_cls=FAISS,
    # 结果条数
    k=1,
)

# 使用小样本提示词模版
similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入词的反义词",
    suffix="原词：{adjective}\n反义：",
    input_variables=["adjective"],
)

pprint('输入短的')
pprint(similar_prompt.format(adjective="big"))