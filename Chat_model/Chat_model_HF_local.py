from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
import os

import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Set Hugging Face cache directory to current working directory
os.environ['HF_HOME'] = os.getcwd()
os.environ['TRANSFORMERS_HOMe'] = os.getcwd()  # Optional, extra safe

llm = HuggingFacePipeline.from_model_id(
    model_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=50
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)


# not exaclty practical but just to show how to use it. will cover it if we need to use it in future.