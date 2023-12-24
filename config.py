import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv


def configure_genai_and_streamlit():
    load_dotenv()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    st.set_page_config(page_title="TTT", page_icon="microphone")
