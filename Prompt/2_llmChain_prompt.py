# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()

# # Step 1: Instantiate LangChain's HuggingFacePipeline with a model ID directly
# llm = HuggingFacePipeline.from_model_id(
#     model_id="google/flan-t5-small",
#     task="text2text-generation",
#     device=0,  # set -1 for CPU
#     model_kwargs={"temperature": 0.7,},
# )
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="conversational",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)

# Step 2: Create a PromptTemplate with variables
template = """
You are a helpful assistant.

Explain the following question in {style} style:

Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["question", "style"])

# Step 3: Combine prompt + model in a chain
chain = prompt | model

# Step 4: Streamlit UI
st.title("LangChain HuggingFace Research Tool")

question = st.text_input("Enter your question", "What is the capital of France?")
style = st.selectbox("Explanation style", ["Beginner-Friendly", "Technical", "Concise"])

if st.button("Submit"):
    with st.spinner("Generating response..."):
        output = chain.invoke({"question": question, "style": style})
        st.write(output.content)


# from langchain_huggingface.chat_models import ChatHuggingFace
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv
# import streamlit as st
# import os

# # Load environment variables
# load_dotenv()

# # Step 1: Initialize the HF Endpoint
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-alpha",
#     task="conversational",  # this model only supports this task
# )

# model = ChatHuggingFace(llm=llm)

# # Step 2: Build the prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Answer in {style} style."),
#     ("human", "{question}")
# ])

# # Step 3: Streamlit UI
# st.title("LangChain Chat using HuggingFace Zephyr")

# question = st.text_input("Enter your question", "What is the capital of France?")
# style = st.selectbox("Explanation style", ["Beginner-Friendly", "Technical", "Concise"])

# if st.button("Submit"):
#     with st.spinner("Generating response..."):
#         chain = prompt | model
#         output = chain.invoke({"question": question, "style": style})
#         st.write(output.content)
