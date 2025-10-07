import streamlit as st
import os

st.set_page_config(page_title="SQLite Streamlit App", layout="wide")


st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Welcome", "Upload Existing DB", "Build DB from CSV"])

if page == "Welcome":
    import landing_page
    landing_page.app()
elif page == "Upload Existing DB":
    import upload_db
    upload_db.app()
elif page == "Build DB from CSV":
    import build_db
    build_db.app()