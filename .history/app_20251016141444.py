import streamlit as st
import subprocess


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Tempora | Time Series Analysis App",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Welcome", "Upload", "Transform", "Explore", "Forecast", "Future"]
)

# -------------------------------
# Page routing logic
# -------------------------------
if page == "Welcome":
    st.title("Welcome to the Time Series Analysis App")

    st.markdown("""
    ### Overview

    This application provides an interactive environment for **transforming, exploring, and forecasting time series data**.

    To learn more about the underlying methods, visit the [Data Archive](https://rhyslwells.github.io/Data-Archive) for articles related to time series.
                
    ---

    ### How to Use This App

    1. Navigate through sidebar.  
    2. Upload your time series data in `.csv` or `.xlsx` format.  
    3. In the **Transform** section, select a **data frequency**, **date range**, and **interpolate** missing values.
    4. In the **Explore** section, inspect trends, seasonality, and noise components.  
    5. In the **Forecast** section, select a model type, configure parameters, and download forecasts.

    ---

    ### Important Notes

    - Uploaded datasets and model outputs exist **only for the current session** and will be lost when you refresh or close the app.  
    - Ensure your data includes a **date column** (daily, weekly, monthly) and a **target variable** for best results.  
    - Large datasets or extensive grid searches may increase computation time.  

    For source code and examples, see the [time-series-analysis repository](https://github.com/rhyslwells/TimeSeries-App).
    """)

def get_last_commit_date():
    try:
        result = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=iso"], stderr=subprocess.DEVNULL
        )
        return result.decode("utf-8").strip()
    except Exception:
        return "Unknown (Git not available)"

last_commit_date = get_last_commit_date()
st.caption(f"**App last updated:** {last_commit_date}")


elif page == "Upload":
    from views import upload
    upload.app()

elif page == "Transform":
    from views import transform
    transform.app()

elif page == "Explore":
    from views import explore
    explore.app()

elif page == "Forecast":
    from views import forecast
    forecast.app()

elif page == "Future":
    from views import future
    future.app()
