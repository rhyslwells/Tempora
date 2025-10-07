# app.py
import streamlit as st
from pages import*

st.set_page_config(page_title="Time Series App", layout="wide")
pages.navigation.run()
