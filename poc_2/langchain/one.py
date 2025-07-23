from langchain_ollama import OllamaLLM
from langchain import PromptTemplate

template = "Come up with names for a {object}"

prompt = PromptTemplate(
    input_variables = ["object"],
    template = template,
)
response = prompt.format(object = "baby boy")

model = OllamaLLM(model="llama3")
print(model.invoke(response))