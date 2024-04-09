import streamlit as st
from clarifai.modules.css import ClarifaiStreamlitCSS
from utils import *


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
    "<h1 style='text-align: center; color: black;'> Upload your data 📂</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=180)"
)

st.markdown("**upload your files here**")
uploaded_files = st.file_uploader(
                "_", accept_multiple_files=True, label_visibility="hidden")

if uploaded_files:
    with st.spinner("Processing the tables..."):
        elem = process_pdf(uploaded_files)
        if 'few_shot_examples' not in st.session_state.keys():
            st.session_state['few_shot_examples'] = elem
        

