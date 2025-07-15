# 使用 LCEI 支持在运行的时候对链进行配置
# 动态改写模型自由度（temperature）
# 模型自由度，0为最确定（根据输入生成最可能的输出），1为最随机（根据输入生成最不相关的输出【更有创意】）0.7为一个阈值
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import ConfigurableField

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
).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM temperature",
        # LLM模型的温度。0是最确定的，1是最随机的。0.7是一个很好的阈值。
        description="The temperature of the LLM model. 0 is the most deterministic, 1 is the most random. 0.7 is a good threshold.",
    )
)


response = llm.invoke("请写一个简短故事的开头")
print(response)

# 使用 LCEI 支持在运行的时候对链进行配置
response2 = llm.with_config(configurable={"llm_temperature": 0.9}).invoke("请写一个简短故事的开头")
print(response2)