from langchain_core.prompts import PromptTemplate

template_str = """
You are a helpful assistant.

Question: {question}

Please answer concisely.
"""

prompt = PromptTemplate(template=template_str, input_variables=["question"])

# To get the filled prompt:
filled_prompt = prompt.format(question="What is the capital of France?")
print(filled_prompt)
