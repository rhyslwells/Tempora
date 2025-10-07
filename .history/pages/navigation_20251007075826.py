# structure/navigation.py
import streamlit as st

def run():
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Welcome", "Upload", "Transform", "Explore", "Forecast"]
    )

    if page == "Welcome":
        from structure import landing_page
        landing_page.app()

    elif page == "Upload":
        from structure import upload
        upload.app()

    elif page == "Transform":
        from structure import transform
        transform.app()

    elif page == "Explore":
        from structure import explore
        explore.app()

    elif page == "Forecast":
        from structure import forecast
        forecast.app()
