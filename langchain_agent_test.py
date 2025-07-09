import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool

load_dotenv(verbose=True, override=True)
# deepseek-r1 模型
# llm = ChatDeepSeek(
#     model="deepseek-reasoner",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     # 最大重试次数
#     max_retries=2,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_base=os.getenv("OPENAI_API_BASE"),
# )

# deepseek-chat 模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    # 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出【更有创意】）0.7为一个阈值
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)

def calculator(input_str):
    try:
        return str(eval(input_str))
    except Exception as e:
        return str(e)


calculator_tool = Tool(
    name = 'Calculator',
    func = calculator,
    description = "用来执行数据计算，输入如 '2 + 2'",
)

agent = initialize_agent(
    tools = [calculator_tool],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
)

agent.run('计算一下 123 乘 456 是多少')
