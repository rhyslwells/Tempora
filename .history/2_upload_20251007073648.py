import streamlit as st
import sqlite3
import datetime
import pandas as pd
import os
import tempfile

from utils.visualization import generate_mermaid_er
import streamlit_mermaid
from utils.db_utils import create_connection

def get_timestamped_db_name():
    return f"db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def app():
    st.title("Upload and Query SQLite DB in Memory")

    # --- Initialization ---
    if "conn" not in st.session_state:
        st.session_state.conn = None
    if "sample_loaded" not in st.session_state:
        st.session_state.sample_loaded = False

    # --- Load Sample DB Button ---
    st.subheader("Try a Sample Database")
    st.markdown("""
    The sample database contains the following:
    - Multiple related tables
    - Example data to test ER diagrams and queries
    - File: `longlist.db` from [CS50’s Introduction to Databases with SQL](https://cs50.harvard.edu/sql/2024/)
    """)

    if st.button("Load Example Database"):
        example_path = os.path.join("sample_data", "longlist.db")
        if os.path.exists(example_path):
            with open(example_path, "rb") as f:
                db_bytes = f.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                tmp_file.write(db_bytes)
                tmp_filename = tmp_file.name

            disk_conn = sqlite3.connect(tmp_filename, check_same_thread=False)
            mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            disk_conn.backup(mem_conn)
            disk_conn.close()
            os.remove(tmp_filename)

            st.session_state.conn = mem_conn
            st.session_state.sample_loaded = True
            st.success("Example database loaded into memory.")
        else:
            st.error("Sample database not found.")

