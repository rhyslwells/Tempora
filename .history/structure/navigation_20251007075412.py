import streamlit as st
import os
im

st.set_page_config(page_title="Time Series App", layout="wide")


st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Welcome", "Upload Existing DB", "Build DB from CSV"])

# pages: ["Welcome", "Upload","Transform","Explore","Forecast"]

if page == "Welcome":
    import landing_page
    landing_page.app()
elif page == "Upload":
    import upload
    upload.app()
elif page == "Transform":
    import transform
    transform.app()
elif page == "Explore":
    import explore
    explore.app()
elif page == "Forecast":
    import forecast
    forecast.app()

