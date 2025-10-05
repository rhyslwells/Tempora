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


# file_uploads.py

def upload_sql_schema():
    uploaded_file = st.file_uploader("Upload a .sql schema file", type=["sql"])
    if uploaded_file is not None:
        try:
            # Read raw bytes and decode
            content = uploaded_file.getvalue().decode("utf-8")
            if content.strip():
                st.text_area("Preview Schema SQL", content, height=200)
                return content
            else:
                st.warning("Uploaded SQL file is empty.")
        except Exception as e:
            st.error(f"Could not decode SQL file: {e}")
    return None
