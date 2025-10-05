import streamlit as st
import pandas as pd

def upload_csv_files():
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    return uploaded_files

def upload_sqlite_db():
    uploaded_db = st.file_uploader("Upload SQLite DB file (.db)", type=['db'])
    if uploaded_db is not None:
        return uploaded_db  # Return the file-like object directly
    return None

