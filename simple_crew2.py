import crewai
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool

llm = Ollama(model="openhermes")

@tool("DuckDuckGoSearch")
def search(search_query: str) -> str:
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

@tool("websearch")
def tool_websearch(q: str) -> str:
    """DuckDuckGo broswer"""
    return DuckDuckGoSearchRun().run(q)

@tool("amazon")
def tool_amazon(q: str) -> str:
    """Search Amazon"""
    return DuckDuckGoSearchRun().run(f"site:amazon.com {q}")



def callback_function(output):
    print(f"Task completed: {output.raw_output}")

prompt = '''Find a laptop with good reviews for less than $1,000.00'''

agent_amazon_search = crewai.Agent(
    role="Amazon Searcher",
    goal="Find a good deal on a new laptop",
    backstory="As a savvy comparison shopper, you need to find a good deal on a laptop for less than $1,000.00",
    tools=[search],
    llm=llm,
    allow_delegation=False, verbose=True)

task_amazon = crewai.Task(description=prompt,
                   agent=agent_amazon_search,
                   expected_output='''Laptop that you chose and explain why it is a good deal''')

crew = crewai.Crew(agents=[agent_amazon_search], tasks=[task_amazon], verbose=True)
res = crew.kickoff()
print(res)
