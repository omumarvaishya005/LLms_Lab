from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Set up the LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="conversational",  # required by this model
)

model = ChatHuggingFace(llm=llm)

# Load prompt from file
prompt = load_prompt("template.json")

# Create a chain using the prompt and model
chain = LLMChain(llm=model, prompt=prompt)

# Define input data
input_data = {
    "question": "How do I train a neural network?",
    "style": "professional"
}

# Run the chain
response = chain.invoke(input_data)

# Output result
print("Response:\n", response["text"])
