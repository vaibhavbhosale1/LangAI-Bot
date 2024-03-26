## Integrate our code with OPenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI


import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit framework

st.title("Vaibhav's Personal ChatBot")
input_text=st.text_input("Search the topic you want")

## OPENAI LLMS
llm=OpenAI(temperature=0.8) # Control Agent



if input_text:
    st.write(llm(input_text))