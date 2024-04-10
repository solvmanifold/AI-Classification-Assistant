import streamlit as st
from streamlit_option_menu import option_menu
from utils import *
import pandas as pd
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain_community.llms import Clarifai as ClarifaiLLM

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] p {
        font-size: 24px;
        font-weight: bold;
    }
    .st-cd {
        gap: 3rem;
    }
</style>""", unsafe_allow_html=True)

st.markdown(
  "<h1 style='text-align: center; color: black;'> Classify your data 🔍</h1>",
  unsafe_allow_html=True,
)
st.markdown(
    "![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=180)"
)

if 'start_chat' not in st.session_state.keys():
    st.session_state['start_chat'] = False
    
if "chat_history" not in st.session_state.keys():
  st.session_state['chat_history'] = [{"Query": "Let's classify", "content": "How may I help you?"}]

with st.sidebar:
  PAT = st.text_input("Enter your Clarifai PAT", type="password", key="PAT")
  model_url = st.text_input("Enter Clarifai LLM model URL", key="model_url")
  if model_url:
    llm = ClarifaiLLM(model_url=model_url, pat=PAT)
    if "config" not in st.session_state.keys():
      st.session_state["config"] = True


def chatbot():
  """chatbot """
  if message := st.chat_input(key="input"):
    st.chat_message("user").write(message)
    st.session_state['chat_history'].append({"role": "user", "content": message})
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        PT = prompt_template(st.session_state['few_shot_examples'], message)
        response = llm(PT)
        st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state['chat_history'].append(message)
        
def selectbox():
    with st.sidebar:
        counter1 = option_menu("previous queries", [st.session_state['chat_history'][-1],st.session_state['chat_history'][-2]],icons=['chat', 'chat'], menu_icon="cast", default_index=0)
        if counter1:
            st.write(str(counter1))
            
def textbox():
   user_query = st.text_area("Enter your text here", key="text_area")
   if user_query:
    with st.spinner("Thinking..."):
          PT = prompt_template(st.session_state['few_shot_examples'], user_query)
          response = llm(PT)
          st.write(response)
          message = [{"Query": user_query, "Response": response}]
          st.session_state['chat_history'].append(message)
          if len(st.session_state['chat_history']) >= 2:
            selectbox()
        
if not st.session_state['start_chat']:
    chatbtn = st.button("Start Chatting", key="chat_btn")
    if chatbtn:
       st.session_state['start_chat'] = True
       st.experimental_rerun()
try:
  if st.session_state['start_chat'] and st.session_state["config"] and st.session_state['few_shot_examples']:
    textbox()
    st.markdown(
        "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>",
        unsafe_allow_html=True,
    )
except Exception as e:
  st.error(f"""Classification failed due to {e},\n\n1. Make sure you have uploaded the data\n2. Make sure you have entered the Clarifai PAT\n3. Make sure you have entered the Clarifai LLM model URL""")
