import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import difflib

groq_api_key = st.secrets("GROQ_API")
client = Groq(api_key=groq_api_key)

# Load FAQ from file
def load_faq(file_path="faq.txt"):
    faq_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            if "|" in line:
                question, answer = line.strip().split("|", 1)
                faq_dict[question.lower()] = answer
    return faq_dict

faq = load_faq()

# Fuzzy match user input to FAQ questions
def check_faq(user_input):
    normalized = user_input.strip().lower()
    match = difflib.get_close_matches(normalized, faq.keys(), n=1, cutoff=0.8)
    if match:
        return faq[match[0]]
    return None

# Streamlit UI
st.set_page_config(page_title="Finance Advisor Chatbot")
st.title("💰 AI Financial Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful financial advisor. Give simple, beginner-friendly answers. Avoid tax or legal advice."}
    ]

# Display chat history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a financial question...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    faq_answer = check_faq(user_input)

    if faq_answer:
        # Respond from FAQ
        with st.chat_message("assistant"):
            st.markdown(faq_answer)
        st.session_state.messages.append({"role": "assistant", "content": faq_answer})
    else:
        # Fallback to Groq model
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=st.session_state.messages,
                temperature=0.5,
                stream=True,
            )
            assistant_response = ""
            response_container = st.empty()
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    assistant_response += delta
                    response_container.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
