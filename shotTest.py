# zeroshot 会导致低质量回答 不给示例，直接回答
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

load_dotenv(verbose=True, override=True)
# deepseek-chat 模型
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
question = "什么是 2 🐦 9？"
response = llm.invoke(question)
pprint(response)

# 增加示例

# 增加示例组
examples = [
    { "input": "2 🐦 2", "output" : "4"},
    { "input": "2 🐦 3", "output" : "5"},
]

# 构造提示词模版
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 组合示例与提示词
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 打印提示词模版
pprint("组合之后的提示词--------------------------")
pprint(few_shot_prompt.invoke({}).to_messages())

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位神奇的数学奇才"),
    few_shot_prompt,
    ("human", "{input}"),
])

# 重新提问
chain = final_prompt | llm
response = chain.invoke({"input": question})
pprint("处理后的答案--------------------------")
pprint(response)