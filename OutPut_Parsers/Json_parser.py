from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me name, age and city about a friction persion  \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt=template.format()
result= model.invoke(prompt)    

print(result)
print(result.content)

# we can chain it to get the results 


# we need to add extra Parser to get the output in json format
chain = template | model 
chain1 = template | model | parser

result = chain.invoke({})
result1 = chain1.invoke({})

print(result)
print(result1)


# now problem with json parser is that it will json object but not exactly the way we want it may be it will give us json but but in formate we can use 



template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print("--- Result with JsonOutputParser --- ")
print(result)
# output 
{'facts': ['Black holes are regions in space where gravity is so strong that nothing, not even light, can escape.', 
           'They form when massive stars die and collapse under their own gravity.', 
           'The event horizon is the boundary around a black hole beyond which escape is impossible.', 
           'Black holes can exert powerful gravitational forces that can distort spacetime around them.', 
           'Scientists are still learning about black holes, and they continue to generate new discoveries.']}

# but may be we needed like this 
{
    'fact1': 'Black holes are regions in space where gravity is so strong that nothing, not even light, can escape.',
    'fact2': 'They form when massive stars die and collapse under their own gravity.',
    'fact3': 'The event horizon is the boundary around a black hole beyond which escape is impossible.',
    'fact4': 'Black holes can exert powerful gravitational forces that can distort spacetime around them.',
    'fact5': 'Scientists are still learning about black holes, and they continue to generate new discoveries.'
}

# so here JsonOutputParser fails to give us the output in the way we want it .