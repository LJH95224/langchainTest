from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from pprint import pprint

prompt_template = ChatPromptTemplate(
    [
        ("system", "你是一个厉害的人工智能助手"),
        MessagesPlaceholder("msgs")
    ]
)

response = prompt_template.invoke({"msgs": [HumanMessage(content="hi! 你好")]})
pprint(response)

"""
ChatPromptValue(
    messages=[SystemMessage(content='你是一个厉害的人工智能助手', additional_kwargs={}, response_metadata={}), 
    HumanMessage(content='hi! 你好', additional_kwargs={}, re={})]
)
"""