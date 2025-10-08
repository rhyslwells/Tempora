import streamlit as st

st.set_page_config(
    page_title="Time Series Analysis App",
    page_icon="📈",
    layout="wide"
)

st.title("Welcome to the Time Series Analysis App")

st.markdown("""
### Overview

This app lets you explore, decompose, and forecast time series data
using interactive visualizations and model selection tools.
""")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Welcome", "Upload", "Transform", "Explore", "Forecast"]
)