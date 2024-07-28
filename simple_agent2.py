import crewai
from langchain_community.llms import Ollama
from crewai_tools import tool


llm = Ollama(model="mistral")

# Define custom search tool
@tool
def search_web(query):
  '''Search web'''
  # Implement your web search logic here (e.g., using Google Search API)
  # For simplicity, let's simulate a search result
  return f"Search result for '{query}': Some relevant information"

# Define agents
#researcher = crewai.Agent(name="Researcher", tools=[search_web], llm=llm)
researcher = crewai.Agent(
    role="Calendar",
    goal="research the provided topic",
    backstory="You are a researcher. You provide info about research topics ",
    tools=[search_web],
    llm=llm,
    allow_delegation=False, verbose=True)
writer = crewai.Agent(
    role="writer",
    goal="summarize the info",
    backstory="You are a summarizer. You provide summaries about research topics ",
    tools=[search_web],
    llm=llm,
    allow_delegation=False, verbose=True)
# writer = crewai.Agent(name="Writer", llm=llm)

# Define task
def summarize_info(topic):
  # Researcher gathers information
  info = researcher.run_tool("search_web", query=topic)
  # Writer summarizes the information
  summary = writer.ask(f"Summarize this information: {info}")
  return summary

# Create and run the Crew
task = crewai.Task(description="Summarize a topic", function=summarize_info, args=["AI"],expected_output="summary")
crew = crewai.Crew(agents=[researcher, writer], tasks=[task])
result = crew.run()

print(f"Summary: {result}")