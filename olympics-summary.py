from langchain_community.llms import Ollama
 
llm = Ollama(model="openhermes")
res = llm.invoke(input=["Summarize 3 highlights from the 2024 Paris Olympics"]).split("\n")[0]
print(res)