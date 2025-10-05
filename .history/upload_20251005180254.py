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

    # --- Upload DB ---
    st.subheader("Upload a SQLite Database")
    if not st.session_state.sample_loaded:
        uploaded_db = st.file_uploader("Upload SQLite DB file (.db)", type=['db'], disabled=st.session_state.sample_loaded)
        if uploaded_db:
            uploaded_bytes = uploaded_db.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                tmp_file.write(uploaded_bytes)
                tmp_filename = tmp_file.name

            disk_conn = sqlite3.connect(tmp_filename, check_same_thread=False)
            mem_conn = sqlite3.connect(':memory:', check_same_thread=False)
            disk_conn.backup(mem_conn)
            disk_conn.close()
            os.remove(tmp_filename)

            st.session_state.conn = mem_conn
            st.success("Database loaded into memory successfully")

    # --- Query + ER ---
    if st.session_state.conn:
        st.subheader("Query Interface")

        # --- Example Queries ---
        example_queries = {
            "Show all tables": "SELECT name FROM sqlite_master WHERE type='table';",
            "Books with avg rating > 4.0":
            """
            SELECT title, ROUND(AVG(rating), 2) AS "average rating"
            FROM ratings
            JOIN books ON books.id = ratings.book_id
            GROUP BY book_id
            HAVING "average rating" > 4.0;
            """,
            "Books by publisher 'MacLehose Press' (subquery)":
            """
            SELECT title FROM books WHERE publisher_id = (
                SELECT id FROM publishers WHERE publisher = 'MacLehose Press'
            );
            """,
            "Find publisher ID for 'MacLehose Press'":
            "SELECT id FROM publishers WHERE publisher = 'MacLehose Press';"
        }

        selected_example = st.selectbox("Choose an example query", list(example_queries.keys()))
        query = st.text_area("Enter SQL query", value=example_queries[selected_example], height=120)

        if st.button("Run Query", key="run_query"):
            try:
                df = pd.read_sql_query(query, st.session_state.conn)
                st.dataframe(df)
                st.success(f"Returned {len(df)} rows")
            except Exception as e:
                st.error(f"Query error: {e}")

        if st.button("Show ER Diagram", key="show_er"):
            try:
                mermaid_code = generate_mermaid_er(st.session_state.conn)
                streamlit_mermaid.st_mermaid(mermaid_code)
            except Exception as e:
                st.error(f"Failed to generate ER diagram: {e}")

        if st.button("Download Database", key="download_db"):
            filename = get_timestamped_db_name()
            disk_conn = create_connection(filename)
            st.session_state.conn.backup(disk_conn)
            disk_conn.close()

            with open(filename, "rb") as f:
                data = f.read()
            st.download_button("Download SQLite DB", data=data, file_name=filename, mime="application/x-sqlite3")
            os.remove(filename)
