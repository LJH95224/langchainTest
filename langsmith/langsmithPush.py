from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  # 确保 .env 文件生效
# Connect to the LangSmith client

client = Client()

# Define the prompt

prompt = ChatPromptTemplate([
("system", "You are a helpful chatbot."),
("user", "{question}"),
])

# Push the prompt

print( client.push_prompt("my-prompt", object=prompt))