# app.py
import streamlit as st
from structure import navigation

st.set_page_config(page_title="Time Series App", layout="wide")
navigation.run()
