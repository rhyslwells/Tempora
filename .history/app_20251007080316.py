# app.py
import streamlit as st
from pages import navigation
from pages import landing_page
from pages import upload
from pages import transform
from pages import explore
from pages import forecast

st.set_page_config(page_title="Time Series App", layout="wide")
navigation.run()
