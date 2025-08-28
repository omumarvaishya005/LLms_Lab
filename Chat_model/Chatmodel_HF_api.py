# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
# from dotenv import load_dotenv
# load_dotenv()

# # create a chat model using HuggingFace
# # You can use a specific model or an endpoint
# llm= HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1-7B-Chat-v1.0",
#     task="text-generation",
# ) 

# # now you can pass the model we created by coping the path of repo_id
# model = ChatHuggingFace(model=llm, temperature=0.7)


# result = model.invoke("What is the capital of India")

# print(result.content)


from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation"
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result)


# from langchain_huggingface.chat_models import ChatHuggingFace
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# # import os

# # load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-alpha",  # Or another chat-compatible model
#     task="text-generation",  # or "text2text-generation" if needed
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# model = ChatHuggingFace(llm=llm, temperature=0.7)
# result = model.invoke("What is the capital of India?")
# print(result)



