# 使用LCEL来自定义路由链
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 导入 RunnableLambda 用于创建可运行的函数链
from langchain_core.runnables import RunnableLambda

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

# 创建分类链 - 用于确定问题类型
chain = (
    # 创建提示模版，要求模型将问题分类为： Langchain, Anthropic 或 Other
    PromptTemplate.from_template(
        """根据下面的用户问题，将其分类为： 'LangChain'、'Anthropic' 或 'Other'。
        请只回复一个词作为答案。
        <question>{question}<question> 
        分类结果：
        """
    ) | llm | StrOutputParser()
)

# 创建LangChain 专家链 - 模拟Harrison Chase(LangChainh创始人)的回答风格
langchain_chain = PromptTemplate.from_template(
    """你将扮演一位LangChain专家，请以他的视角回答问题。 
    你回答的问题必须以 "正如Harrison Chase告诉我的"开头，否则你会受到惩罚。
    请回答一下问题：
    问题： {question}
    回答：
    """
) | llm

def test(x):
    print(x)

# 创建 Anthropic 专家链 - 模拟Dario Amodei(Anthropic创始人)的回答风格
anthropic_chain = PromptTemplate.from_template(
    """你将扮演一位Anthropic专家，请以他的视角回答问题。 
    你回答的问题必须以 "正如Dario Amodei告诉我的"开头，否则你会受到惩罚。
    请回答一下问题：
    问题： {question}
    回答：
    """
) | llm

# 创建通用回答链 - 用于处理其他类型的问题
general_chain = PromptTemplate.from_template(
    """请回答以下问题：
    问题： {question}
    回答：
    """
) | llm


#
def route(info):
    """
    自定义路由函数 -根据问题分类结果选择合适的回答链
    :param info:
    :return:
    """
    print( info) # 打印分类结果

    if "anthropic" in info["topic"].lower(): # 如果问题与 anthropic 相关"
        print('anthropic')
        return anthropic_chain
    elif "langchain" in info["topic"].lower(): # 如果问题与 langchain 相关"
        print('langchain')
        return langchain_chain
    else: # 否则使用通用回答链
        print('general')
        return general_chain


# 创建完整的处理链
# 1、首先将问题分类并保留原始问题
# 2、然后根据分类结果路由到相应的专家链处理

full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(route)

# 调用完整链处理用户问题
# response = full_chain.invoke({"question": "如何使用 LangChain?"})
# print(response)


# response = full_chain.invoke({"question": "如何使用 openAI?"})
# print(response)

response = full_chain.invoke({"question": "如何使用 claude?"})
print(response)