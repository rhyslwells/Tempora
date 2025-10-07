# app.py
import streamlit as st
from pages import navigation

st.set_page_config(page_title="Time Series App", layout="wide")
navigation.run()
