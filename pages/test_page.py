import streamlit as st
import pandas as pd
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain_community.llms import Clarifai as ClarifaiLLM
from streamlit_option_menu import option_menu


if "chat_history" not in st.session_state.keys():
  st.session_state['chat_history'] = [{"Query": "Let's classify", "content": "How may I help you?"}]
  
def textbox():
   user_query = st.text_area("Enter your text here", key="text_area")
   if user_query:
    with st.spinner("Thinking..."):
          response = "user asked: " + user_query
          st.write(response)
          message = [{"Query": user_query, "Response": response}]
          st.session_state['chat_history'].append(message)
          if len(st.session_state['chat_history']) >= 2:
            selectbox()

def selectbox():
    with st.sidebar:
        counter1 = option_menu("previous queries", [st.session_state['chat_history'][-1],st.session_state['chat_history'][-2]],icons=['chat', 'chat'], menu_icon="cast", default_index=0)
        if counter1:
            st.write(str(counter1))            

textbox()
            