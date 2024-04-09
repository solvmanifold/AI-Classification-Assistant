import streamlit as st
from utils import *
from clarifai.modules.css import ClarifaiStreamlitCSS
import pandas as pd

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


if 'start_chat' not in st.session_state.keys():
    st.session_state['start_chat'] = False
    
if "chat_history" not in st.session_state.keys():
  st.session_state['chat_history'] = [{"role": "assistant", "content": "How may I help you?"}]


def show_previous_chats():
  for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
      st.write(message["content"])

FSPT = few_shot_template(st.session_state['few_shot_examples'])
st.write(FSPT)

def chatbot():
  ""



if st.session_state['start_chat']:
    show_previous_chats()
    chatbot()
    st.markdown(
        "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>",
        unsafe_allow_html=True,
    )
