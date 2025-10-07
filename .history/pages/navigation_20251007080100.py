# structure/navigation.py
import streamlit as st

def run():
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Welcome", "Upload", "Transform", "Explore", "Forecast"]
    )

    if page == "Welcome":
        from pages import landing_page
        landing_page.app()

    elif page == "Upload":
        from pages import upload
        upload.app()

    elif page == "Transform":
        from pages import transform
        transform.app()

    elif page == "Explore":
        from pages import explore
        explore.app()

    elif page == "Forecast":
        from pages import forecast
        forecast.app()
