# 部分格式化提示词模版
from pprint import pprint
from datetime import datetime
from langchain_core.prompts import PromptTemplate

def _get_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")




prompt = PromptTemplate(
    template="Tell me {adjective} joke about the day {date}",
    input_variables=['adjective', 'date'],
)
partial_prompt = prompt.partial(date=_get_datetime)
pprint(partial_prompt.format(adjective="funny"))

# 'Tell me funny joke about the day 2025-07-11 11:16:40'