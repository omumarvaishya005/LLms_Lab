from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"

from langchain_community.document_loaders import WebBaseLoader
load_dotenv()
# stablish huggingface endpoint

llm =HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="test-generation"
)

model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)
docs = loader.load()
print(type(docs))
# print(docs)

prompt = PromptTemplate(
    template ="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=['question ', 'text']

)

chain= prompt | model | parser
result =  chain.invoke({'question': "what is the product name", 'text': docs[0].page_content})
# print(result)


# from langchain_community.document_loaders import WebBaseLoader
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableLambda,RunnableBranch
# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)


# prompt = PromptTemplate(
#     template='Answer the following question \n {question} from the following text - \n {text}',
#     input_variables=['question','text']
# )

# parser = StrOutputParser()

# url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
# loader = WebBaseLoader(url)

# docs = loader.load()


# chain = prompt | model | parser

# print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))