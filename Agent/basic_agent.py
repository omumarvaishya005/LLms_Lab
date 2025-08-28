from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()


# first of all we need an llm for resoning and planing purpose we are using hugging face model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation",
    temperature=0,
    max_new_tokens=512,
    model_kwargs={
        "stop": ["\nObservation:", "\nAction:"],
        "instructions": (
            "You are a ReAct agent. Only use the following tools: duckduckgo_search. "
            "Always format outputs as: Thought: <text>\nAction: duckduckgo_search\nAction Input: <query>"
        )
    }
)


llm = ChatHuggingFace(llm=llm)

# print(llm.invoke("capital of india"))
# now we need to import these
# 1 from langchain_core.tools import tool ----> for having tools
# 2 import requests ----> for making internet requests 

from langchain_core.tools import tool 
import requests

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# now we need to import neccesary libraries for agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


#Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

#Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True  # retry if parsing fails
)

# Now we can use the agent executor to run our agent

result=agent_executor.invoke({"input":"Explain trump tarrif on india and how it affects indian economy and what is the current status of this tarrif"})
print(result['output'])
print(result)