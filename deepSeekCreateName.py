# 起名，输出格式为一个数组list
import os
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 加载 .env 环境变量文件
# verbose=True 打印加载日志（比如成功加载了哪些变量），有助于调试。
# override=True如果系统环境变量中已存在相同名称的变量，是否覆盖原有的值。默认为 False。
startTime = time.time() * 1000  # 时间毫秒数
load_dotenv(verbose=True, override=True)


# 自定义类，格式化输出
class CommaSeparatedListOutputParser(BaseOutputParser):
    """将LLM调用的输出解析为逗号分隔的列表."""

    def parse(self, text: str):
        """解析LLM调用的输出."""
        # text.strip()：去除字符串首尾的空白符。
        # .split(", ")：按 ", " 分割字符串，返回列表。
        # print("LLM调用的原文输出：", text)
        return text.strip().split(", ")


llm = ChatDeepSeek(
    # model="deepseek-reasoner",
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    # 最大重试次数
    max_retries=2,
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    api_base=os.getenv("DEEP_SEEK_API_BASE"),
)

# prompt 1
# prompt = PromptTemplate.from_template("你是一个起名大师，请模仿示例起男女各取3个{country}名字，比如男孩经常被叫做{boy}, 女孩经常被叫做{girl}。请返回以逗号分隔的列表形式。仅返回逗号分隔的列表，不要反悔其他内容。"
# )

# prompt2
# prompt = PromptTemplate.from_template(
#     "你是一个起名大师，请模仿示例起3个具有{country}特色的名字，示例：男孩常用名{boy}，女孩常用名{girl}。 请返回以逗号分隔的列表形式。仅返回逗号分隔的列表，不要返回其他内容。"
# )

# prompt3
prompt = PromptTemplate.from_template(
    "你是一个起名大师，请模仿示例起3个具有{country}特色的名字，示例：男孩常用名{boy}，女孩常用名{girl}。 请返回以逗号分隔的列表形式。仅返回逗号分隔的列表，不要返回其他内容。请严格按照示例格式返回"
)

message = prompt.format(country="中国特色的", boy="狗蛋", girl="翠花")
print("发送给LLM的message：", message)
strs = llm.invoke(message)

print("接收到的LLM原文", strs)
response = CommaSeparatedListOutputParser().parse(strs.content)
endTime = time.time() * 1000

runTime = endTime - startTime
print("格式化之后的", response)
print(
    "代码运行了{}毫秒，消耗了{}token".format(
        runTime, strs.usage_metadata.get("total_tokens", 0)
    )
)
