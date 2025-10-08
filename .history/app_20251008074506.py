# app.py
import streamlit as st
from pages import*
import pages

st.set_page_config(page_title="Time Series App", layout="wide")
pages.navigation.run()
