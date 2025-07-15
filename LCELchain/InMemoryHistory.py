# 为链增加记忆功能 短时记忆
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory # 导入聊天历史基类

from langchain_core.messages import BaseMessage, AIMessage # 导入消息基类和AI消息类

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

# 获取会话id为 “1” 的历史记录
history = get_by_session_id("1")

# 添加一条AI消息到历史记录
history.add_message([AIMessage("Hello!")])

# 打印存储的所有历史记录
print(store)