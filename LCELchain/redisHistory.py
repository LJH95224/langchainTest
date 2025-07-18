from langchain_redis import RedisChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)

# 最简单实用
# 初始化 Redis 聊天消息历史记录
# 使用 Redis 存储聊天历史，需要提供会话ID 和 Redis 连接URL
print("初始化 Redis 聊天消息历史记录...", os.getenv("REDIS_URL"))
history = RedisChatMessageHistory(session_id="user_123", redis_url=os.getenv("REDIS_URL"))
# 首先清空历史
history.clear()

# 向历史记录中添加消息
history.add_user_message("你好，AI助手！") # 添加用户消息
history.add_ai_message("你好，我今天能为你提供什么帮助？") # 添加AI回复消息

# 检索并显示历史消息
print("聊天历史:")
for message in history.messages:
    print(f"{type(message).__name__}: {message.content}")