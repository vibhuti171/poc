from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

import streamlit as st
summarise_template = PromptTemplate(
    input_variables = ["text"],
    template = 'give a clear and pointed concise summary of: {text}'
)

model = OllamaLLM(model = "llama3")
chain = LLMChain(
    llm = model,
    prompt = summarise_template,
    verbose = False,
    memory = ConversationBufferMemory()
)


st.title('Summariser')
input_text = st.text_input('Enter text to summarise: ')

if input_text:
    response = chain.invoke({"text": input_text})
    st.write(response)

