import streamlit as st

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Time Series Analysis App",
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

    This application provides a complete environment for **exploring, analysing, and forecasting time series data** directly in your browser.

    The app is organised into focused workflows, allowing you to:

    1. **Explore Data**  
       Upload and visualise your time series data using interactive Plotly charts, moving averages, and decompositions.

    2. **Model Forecasts**  
       Build and compare forecasts using statistical models such as **Exponential Smoothing**, **ARIMA**, and **SARIMA**.

    3. **Evaluate Results**  
       Assess model performance, and export both model parameters and evaluation summaries to CSV.

    To learn more about the underlying methods, visit the [Data Archive](https://rhyslwells.github.io/Data-Archive) for articles on  
    [forecasting methods](https://rhyslwells.github.io/Data-Archive/standardised/Forecasting) and  
    [time series decomposition](https://rhyslwells.github.io/Data-Archive/standardised/Decomposition).

    ---

    ### How to Use This App

    1. Navigate through the tabs or sidebar options to choose a workflow (Exploration, Forecasting, Evaluation).  
    2. Upload your time series data in `.csv` format.  
    3. In the **Exploration** section, inspect trends, seasonality, and noise components.  
    4. In the **Forecasting** section, select a model type and configure or grid-search parameters.  
    5. Choose your **optimisation metric** (e.g., AIC, RMSE, MAE) to find the best-performing configuration.  
    6. View and download model summaries, metrics, and parameter results.

    ---

    ### Use Cases

    - Rapid prototyping of forecasting models.  
    - Educational demonstration of statistical time series techniques.  
    - Exploratory analysis for operational, financial, or environmental datasets.  
    - Comparison of different forecasting methods under consistent evaluation metrics.

    ---

    ### Important Notes

    - Uploaded datasets and model outputs exist **only for the current session** and will be lost when you refresh or close the app.  
    - Ensure your data includes a **datetime column** and a **target variable** for best results.  
    - Large datasets or extensive grid searches may increase computation time.  

    For source code and examples, see the [time-series-analysis repository](https://github.com/rhyslwells/time-series-analysis).
    """)

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

elif page == "Future":
    from pages import future
    future.app()
