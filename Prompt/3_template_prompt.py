from langchain_core.prompts import load_prompt # for loading prompts from a file

from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate # for basic prompts 
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

# Step 2: Create a PromptTemplate with variables
# template = """
# You are a helpful assistant.

# Explain the following question in {style} style:

# Question: {question}
# """
#  this will come from a file

# prompt = PromptTemplate(template=template, input_variables=["question", "style"])
prompt = load_prompt("template.json")



# Step 4: Streamlit UI
st.title("LangChain HuggingFace Research Tool")

question = st.text_input("Enter your question", "What is the capital of France?")
style = st.selectbox("Explanation style", ["Beginner-Friendly", "Technical", "Concise"])
filled_text = prompt.format(question=question, style=style)

if st.button("Submit"):
    with st.spinner("Generating response..."):
        output = model.invoke(filled_text)
        st.write(output.content)

prompt = load_prompt("template.json")
filled_text = prompt.format(question=question, style=style)
print(filled_text)
