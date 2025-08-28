from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import streamlit as st
import torch

model_id = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

llm = HuggingFacePipeline(pipeline=pipe)

st.title("Research Tool")

query = st.text_input("Ask something:", "What is the capital of France?")

if st.button("Submit"):
    with st.spinner("Thinking..."):
        response = llm.invoke(query)
        st.write(response)



# Uncomment for extended summarization (future part)
# paper_input = st.selectbox(
#     "Select Research Paper Name",
#     ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
# )
#
# style_input = st.selectbox(
#     "Select Explanation Style",
#     ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
# )
#
# length_input = st.selectbox(
#     "Select Explanation Length",
#     ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
# )
#
# template = load_prompt('template.json')
#
# if st.button('Summarize'):
#     chain = template | model
#     result = chain.invoke({
#         'paper_input': paper_input,
#         'style_input': style_input,
#         'length_input': length_input
#     })
#     st.write(result.content)
