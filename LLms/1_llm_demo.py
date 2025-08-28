from langchain_openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of France?")
print(result)


# 🧠 For LangChain Hands-on Playground Plan
# We'll explore:
#     LLMs: Simple chat/completion (OpenAI, ChatOpenAI)
#     Prompt Templates: Dynamically generate prompts
#     Chains: LLMChain, SequentialChain
#     Tools & Agents: Let LLMs use tools like calculator, search
#     Memory: Chat history, conversation memory
#     Retrieval (RAG): Use your own PDFs/texts for question answering
#     Document Loaders & Vector Stores
#     Output Parsers: Structured output like JSON, tables, etc.
#     Callbacks (Advanced): Trace and debug LangChain flows
#     Streaming responses: Real-time token output

# Although we will have deep understabding of such features for now we will cover basics of it.

# 1. LLM Demo (Chat + Completion)
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
load_dotenv()

chat_llm = ChatOpenAI(model="gpt-3.5-turbo")
completion_llm = OpenAI(model="gpt-3.5-turbo-instruct")

chatResult=chat_llm.invoke("Tell me a joke")
print(chatResult.content)
completion_llm_result = completion_llm.invoke("Translate to Arabic: Hello, how are you?")
print(completion_llm_result)
print(completion_llm_result.content)
