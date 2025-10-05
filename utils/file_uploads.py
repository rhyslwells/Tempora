import streamlit as st
import pandas as pd

def upload_csv_files():
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    return uploaded_files
