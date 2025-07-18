# 检索器，查询重构

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///chinook.db")


# 使用hub 上预制的提示词
from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# 正确地访问 messages 数组
assert len(query_prompt_template.messages) >= 1

query_prompt_template.messages[0].pretty_print()

import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

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

# 使用LCEL 创建一个简单的SQL查询
from typing_extensions import Annotated, TypedDict

# 定义状态类型
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# 定义输出类型
class QueryOutput(TypedDict):
    """生成的SQL查询。"""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# 定义一个write_query函数
def write_query(state: State):
    """生成的SQL查询用于获取信息。"""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            # 最多返回10条
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


# 得到SQL语句，执行SQL语句，执行SQL语句存在风险
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
def execute_query(state: State):
    """执行SQL查询."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


from langchain_core.runnables import RunnablePassthrough

#  定义链以回答SQL查询中的问题
def answer_question(state: State):
    """根据查询结果，格式化答案。"""
    prompt = f"""基于SQL查询:
    {state["query"]}
    
    SQL查询结果:
    {state["result"]}
    
    回答用户的问题: {state["question"]}
    请提供简洁且内容丰富的回复
    """
    return {"answer": llm.invoke(prompt).content}

# 构建从问题到答案的完整链条
sql_chain = (
    RunnablePassthrough.assign(query=write_query)
    .assign(result=execute_query)
    .assign(answer=answer_question)
)


question="获取销售额最高的5位员工及其销售总额"
response_data = sql_chain.invoke({"question": question})
print(response_data)
print("\n 问题")
print(f"question: {question}\n")
print("\n 生成的SQL")
print(f"response_data['query']: {response_data['query']}\n")
print("\n 执行结果")
print(f"response_data['result']: {response_data['result']}\n")
print("\n 生成的答案")
print(f"response_data['answer']:  {response_data['answer']}\n")