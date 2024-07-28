import crewai
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool

llm = Ollama(model="openhermes")

@tool("DuckDuckGoSearch")
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

@tool("olympics-search")
def olympics_search(search_query: str):
    """Search the olympics website"""
    return DuckDuckGoSearchRun().run(f"site:https://olympics.com/en/paris-2024{search_query}")

@tool("amazon")
def tool_amazon(q: str) -> str:
    """Search Amazon"""
    return DuckDuckGoSearchRun().run(f"site:https://olympics.com/en/paris-2024 {q}")

def callback_function(output):
    print(f"Task completed: {output.raw_output}")

prompt = '''Find headlines about 2024 Paris Olympics'''

agent_olympics_reporter = crewai.Agent(
    role="Olympics Reporter",
    goal="Find headlines about 2024 Paris Olympics",
    backstory="As a reporter, you need to find headlines for the 2024 Paris Olympics",
    tools=[search],
    llm=llm,
    allow_delegation=False, verbose=True)

task_olympics = crewai.Task(description=prompt,
                   agent=agent_olympics_reporter,
                   expected_output='''2024 Paris Olympics headlines''')

crew = crewai.Crew(agents=[agent_olympics_reporter], tasks=[task_olympics], verbose=True)
res = crew.kickoff()
print(res)
