import os
import streamlit as st
from streamlit_option_menu import option_menu
from utils import *
import pandas as pd
from clarifai.modules.css import ClarifaiStreamlitCSS
from langchain_community.llms import Clarifai as ClarifaiLLM
from langchain.vectorstores import Clarifai as clarifaivectorstore

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
      
PAT = st.experimental_get_query_params()["pat"][0]

def init_db(no_of_examples):
  query_params = st.experimental_get_query_params()
  USER_ID = query_params["user_id"][0] if "user_id" in query_params.keys() else None
  APP_ID = query_params["app_id"][0] if "app_id" in query_params.keys() else None
  vectorDB = clarifaivectorstore(
    user_id=USER_ID,
    app_id=APP_ID,
    number_of_docs=no_of_examples*3,
    pat = PAT
  )
  return vectorDB

def retrieve_and_parse_rag(user_query, vectorDB):
  examples = rag_examples(vectorDB, user_query)
  return rag_prompt_template(examples,user_query), examples

def selectbox():
    with st.sidebar:
        counter1 = option_menu("previous queries", 
                               [st.session_state['chat_history'][-1],st.session_state['chat_history'][-2]],
                               icons=['chat', 'chat'], menu_icon="cast", default_index=0)

def model_Select():
  llm_model,llm_params = llm_models()
  selected_model = st.selectbox("Select the LLM model", llm_model)
  if selected_model:
    llm = ClarifaiLLM(model_url=llm_params[selected_model], pat=PAT)

  return llm
 
def process_response(text):
    # Find & remove any text after line containing "Remarks"
    remarks_index = text.find("Remarks")
    if remarks_index != -1:
        newline_index = text.find('\n', remarks_index)
        if newline_index != -1:
            text = text[:newline_index]
    # Format newlines for html rendering
    text = text.replace('\n','<br>')
    return text

def textbox(llm, mode, zero_shot_examples, no_of_examples : int = 1):
   user_query = st.text_area("Enter your text here", key="text_area")
   classify_button = st.button("Classify", key="classify_btn") 
   if classify_button: 
    with st.spinner("Thinking..."):
      rag_examples = None
      if mode == "RAG":
        vectorDB = init_db(no_of_examples)
        prompt, rag_examples = retrieve_and_parse_rag(user_query, vectorDB)
      else:
        prompt = prompt_template(zero_shot_examples, user_query)
        
      response = llm.invoke(prompt)
      print(response.replace('\n','\\n'))
      st.markdown(process_response(response), unsafe_allow_html=True)

      if rag_examples:
         st.write("Closest matching items:")
         st.write(rag_examples)

      message = [{"Query": user_query, "Response": response}]
      st.session_state['chat_history'].append(message)
      if len(st.session_state['chat_history']) >= 2:
        selectbox()
 
lm = model_Select()       
if not st.session_state['start_chat']:
    chatbtn = st.button("Start Chatting", key="chat_btn")
    if chatbtn:
       st.session_state['start_chat'] = True
       st.experimental_rerun()
       
with st.sidebar:
  mode = st.radio("**Select the classification mode**",["ICL zero shot","RAG"] )
  
  if mode == "RAG":
    no_of_examples = st.number_input("Enter the number of examples",min_value=1,max_value=25)
  
  if mode:
    zst = zero_shot_contents()
    if "config" not in st.session_state.keys():
      st.session_state["config"] = True

try:
  if st.session_state['start_chat'] and st.session_state["config"]:
    if mode == "RAG":
      textbox(lm, mode, zst, no_of_examples)
    else:
      textbox(lm, mode, zst)
      
    st.markdown(
        "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>",
        unsafe_allow_html=True,
    )
except Exception as e:
  st.error(f"""Classification failed due to {e}.""")
