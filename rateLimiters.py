# 速率限制
# langchain 中的 InMemoryRateLimiter 只能限制单位时间内的请求数量

import os
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv(verbose=True, override=True)


# 创建了一个内存中的速率限制器（InMemoryRateLimiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 表示每秒最多允许 0.1 个请求，也就是 每 10 秒最多允许 1 次调用。
    check_every_n_seconds=0.1, # 表示速率检查器每 0.1 秒运行一次，检查是否有新令牌（请求配额）可以用。
    max_bucket_size=10 # 令牌桶最大容量是 10。如果你长时间不请求，会“存”下最多 10 次可用请求额度。
)

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
    rate_limiter=rate_limiter,
)

for _ in range(5):
    tic = time.time()
    response = llm.invoke('hello')
    toc = time.time()
    print(toc - tic)
    print(response)