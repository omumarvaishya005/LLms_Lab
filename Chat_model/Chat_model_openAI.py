from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.0)
result=model.invoke("write a small story about india growth in it")
# print(result)
print(result.content)