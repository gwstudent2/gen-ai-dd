import crewai
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool
from crewai_tools import WebsiteSearchTool



def callback_function(output):
    print(f"Task completed: {output.raw_output}")

search_tool = WebsiteSearchTool(
 #   website="https://google.com",
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistral",
                temperature=0.5,
                top_p=1,
                stream=True,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="mistral",
            ), 
        ),
    )
)

agent = crewai.Agent(
    role="Calendar",
    goal="What day of the month is Thanksgiving on in 2024?",
    backstory="You are a calendar assistant. You provide information about dates. ",
    tools=[search],
    llm=llm,
    allow_delegation=False, verbose=True)

task = crewai.Task(description="What day of the month is Thanksgiving on in 2024?",
                   agent=agent,
                   expected_output="Date of Thanksgiving in the current year")

crew = crewai.Crew(agents=[agent], tasks=[task], verbose=True)
res = crew.kickoff()
print(res)
