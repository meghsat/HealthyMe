import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title('Healthy Me!')
input_text=st.text_input("Search the topic u want") 

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about the {name} disease"
)

name_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
disease_memory = ConversationBufferMemory(input_key='disease', memory_key='chat_history')
symptoms_memory = ConversationBufferMemory(input_key='symptoms', memory_key='description_history')

llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='disease',memory=name_memory)



second_input_prompt=PromptTemplate(
    input_variables=['disease'],
    template="What are the symptoms of {disease}"
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='symptoms',memory=disease_memory)

third_input_prompt=PromptTemplate(
    input_variables=['symptoms'],
    template="What medications to use to dimnish these symptoms: {symptoms}"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='Medications',memory=symptoms_memory)
parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['disease','symptoms','Medications'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text})) # when input_text is run the field given here is assigned to the below 'name' variable

    with st.expander('Disease Name'): 
        st.info(disease_memory.buffer)

    with st.expander('Major Symptoms'): 
        st.info(symptoms_memory.buffer)