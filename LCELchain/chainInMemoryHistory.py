import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带历史记录的可运行组件
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory # 导入聊天历史基类

from langchain_core.messages import BaseMessage, AIMessage # 导入消息基类和AI消息类

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

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """在内存中实现的聊天消息历史记录"""

    # 创建一个列表，用于存储消息
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: List[BaseMessage]) -> None:
        """添加一组消息到存储中"""
        self.messages.extend(message)


    def clear(self) -> None:
        """清空存储中的消息"""
        self.messages = []


# 使用全局变量存储聊天消息历史记录
# 创建空字典用于存储不同回话的历史记录
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """根据会话id获取历史记录，如果不存在则创建新的"""
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# 创建聊天提示模版，包含系统提示，历史记录和用户问题
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个擅长{ability}的助手"), # 系统角色提示，使用 ability 参数定义助手专长
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}") # 用户问题占位符，使用 question 参数
])

# 将提示模版与 deepseek 模型连结成一个链
chain = prompt | llm

# 创建带有消息历史功能的可运行链
chain_with_history = RunnableWithMessageHistory(
    chain, # 基础链
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

response = chain_with_history.invoke(
    {"question": "余弦函数时什么意思", "ability": "math"},
    config={"configurable": {"session_id": "foo"}} # 配置回话ID为foo
)

response2 = chain_with_history.invoke(
    {"question": "它的反函数是什么？", "ability": "math"},
    config={"configurable": {"session_id": "foo2"}} # 配置回话ID为foo
)


print(store)