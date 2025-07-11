# 部分格式化提示词模版
from pprint import pprint

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{foo}{bar}")
partial_prompt = prompt.partial(foo="foo-test")
pprint(partial_prompt.format(bar="baz"))

# 'foo-testbaz'
