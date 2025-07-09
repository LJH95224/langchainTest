# 字符模版
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你是一个{name}, 帮我起{num}个具有{county}特色的{sex}名字")
prompts = prompt.format(name="算命大师", num="2", county="中国", sex="男孩")
print(prompts)