from langchain_community.llms import Ollama
 
llm = Ollama(model="openhermes")
res = llm.invoke(input=["What day of the month is Thanksgiving on in 2024? "]).split("\n")[0]
print(res)