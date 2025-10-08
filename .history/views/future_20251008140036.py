import streamlit as st

# ---------------- PAGE SETUP ---------------- #
# st.set_page_config(page_title="Future Improvements", layout="wide")
def app():

    st.title(":orange[Future Additions & Improvements]")
    st.markdown("This page outlines planned enhancements for upcoming versions of the Time Series Analysis app.")

    st.divider()

    # ---------------- MAIN CONTENT ---------------- #
    st.markdown(
        """

    ## **Welcome**
    - [ ] Provide a **“Quick Start Guide”** for first-time users  
    - [ ] Add a **data freshness check** (last updated timestamp)  

    ---

    ## **Transform**
    - [ ] Provide a **“Quick Start Guide”** for first-time users  
    - [ ] Allow **automatic frequency detection** and resampling (daily, monthly, etc.)  
    - [ ] Include a **data cleaning and missing value handling** section  


    ---

    ## **Explore**
    - [ ] Add **Plotly charts** for decomposition, smoothing, and rolling averages  
    - [ ] Include **distribution plots** (histogram or KDE of returns)  
    - [ ] Add **correlation heatmaps** or cross-correlation between columns  
    - [ ] Add **toggle to normalize** or scale data before analysis  
    - [ ] Add **outlier detection** and anomaly marking  


    ---

    ## **Forecast|ARIMA Models Section**
    - [ ] Implement **grid search** for best (p, d, q) parameters automatically  
    - [ ] Add **AIC/BIC comparison table** for multiple ARIMA fits  
    - [ ] Enable **residual diagnostics plots** using Plotly  
    - [ ] Add **forecast accuracy metrics** (MAE, RMSE, MAPE)  
    - [ ] Allow **user selection of forecast horizon** (steps ahead)  
    - [ ] Display **confidence intervals** clearly on plots 

    ---

    ## **Forecast|Prophet Tab**
    - [ ] Allow **custom regressors** or holidays  
    - [ ] Enable **parameter tuning controls** (changepoint, seasonality scale)  
    - [ ] Add **interactive forecast plots** with confidence shading  

    ---

    ## **Forecast|Future Model Expansions**
    - [ ] Integrate **SARIMA / SARIMAX** models  
    - [ ] Add **RNN / LSTM** architectures (user-defined layers and parameters)  
    - [ ] Implement **VAR / VARMAX** for multivariate forecasting  

    ---

    ## **General | Learning / Documentation Features / Experience**
    - [ ] Add ** explanations** for each model type and parameter  
    - [ ] Include **“Learn More” sections** beside outputs
    - [ ] Add **sample use cases** or datasets for quick demos 
    - [ ] Add **progress indicators** for model fitting and grid search  
    - [ ] Add **icons / color indicators** for model performance  
    - [ ] Enable **report generation** (PDF/Markdown summary of results)  

    ---

    ### *Version Roadmap*
    - **v1.1** → Custom dataset upload, Plotly integration  
    - **v1.2** → Grid search and automated ARIMA tuning  
    - **v1.3** → Prophet advanced controls and VAR models  
    - **v1.4** → RNN/LSTM
    ---
        """
    )
