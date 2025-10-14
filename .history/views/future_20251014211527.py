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
        #### **1. Welcome Page**

        * [ ] Add a **Quick Start Guide** for first-time users with sample workflow and demo dataset.
        * [ ] Display **data freshness information** (last updated timestamp from source file or upload metadata).

        ---

        #### **2. Transform Module**

        * [ ] Implement **automatic frequency detection** (daily, weekly, monthly) and resampling options.
        * [ ] Add **data cleaning utilities**:

        * Handle missing values (imputation, interpolation, or forward-fill).
        * Detect and remove duplicate timestamps.
        * [ ] Remove **seasonal differencing**; replace with **general differencing options** (first and second order).
        * [ ] Add **stationarity diagnostics** (ADF test with differencing recommendations).

        ---

        #### **3. Explore Module**

        * [ ] Add **interactive Plotly visualizations**:

        * Rolling mean and standard deviation.
        * Seasonal decomposition (trend, seasonality, residual).
        * [ ] Add **distribution analysis**:

        * Histogram and kernel density plots.
        * Normality tests (Jarque–Bera, Shapiro–Wilk).
        * [ ] Add **residual analysis suite**:

        * Histogram and QQ plot of residuals.
        * ACF plot of residuals (independence check).
        * Statistical tests for heteroskedasticity and autocorrelation.
        * [ ] Add **correlation heatmaps** and **cross-correlation plots** for multivariate data.
        * [ ] Add toggle for **data normalization or scaling** before visualization.
        * [ ] Implement **anomaly detection** using IQR, z-score, or rolling MAD.

        ---

        #### **4. Forecast Module (ARIMA & SARIMA)**

        * [ ] Implement **grid search** for ARIMA/SARIMA parameters:

        * Search over $(p, d, q)$ and seasonal $(P, D, Q, m)$.
        * Filter out non-stationary configurations automatically.
        * [ ] Display **AIC/BIC comparison table** for all fitted models.
        * [ ] Add **residual diagnostics** using tools from the Explore module.
        * [ ] Include **forecast accuracy metrics** (MAE, RMSE, MAPE).
        * [ ] Enable **user selection of forecast horizon** (number of steps ahead).
        * [ ] Add **Plotly-based forecast plot** with shaded confidence intervals.
        * [ ] **Note:** (S)ARMA models assume stationarity; differencing or detrending may be required.

        ---

        #### **5. Forecast Module (Prophet)**

        * [ ] Add **custom regressor** and **holiday/event** support.
        * [ ] Include **parameter tuning controls**:

        * Changepoint prior scale.
        * Seasonality prior scale.
        * Growth type.
        * [ ] Add **interactive forecast plots** with shaded confidence intervals.
        * [ ] Include **cross-validation metrics** (MAPE, RMSE).

        ---

        #### **6. Auto-Forecast Module**

        A fully automated forecasting system that tests, compares, and selects the best model.

        * [ ] Develop an **automated model selection engine**:

        * Compare ARIMA, SARIMA, Prophet, ETS, and naïve baseline models.
        * Rank models based on AIC/BIC and validation metrics (MAPE, RMSE).
        * [ ] Include **automatic stationarity detection and transformation**:

        * Differencing, log-scaling, Box–Cox transformations if required.
        * [ ] Automate **train/test splitting** (e.g., last 20% of data for validation).
        * [ ] Add a **model comparison dashboard**:

        * Display ranked results with metrics and residual diagnostics.
        * Plot top models’ forecasts together for comparison.
        * [ ] Add **hyperparameter optimization** (grid or random search).
        * [ ] Generate **summary reports** with:

        * Best model type and configuration.
        * Key metrics and plots.
        * Suggested model rationale.
        * [ ] (Optional) Implement **meta-learning hints** — recommend model families based on time series characteristics (e.g., strong seasonality → Prophet).

        ---

        #### **7. Documentation & Learning**

        * [ ] Add **inline explanations** for each model type and parameter.
        * [ ] Include **“Learn More” expandable sections** beside model results.
        * [ ] Provide **sample datasets and workflow demos**.
        * [ ] Add **progress indicators** (spinners or progress bars) for model fitting and grid search.
        * [ ] Include **performance indicators** (icons or color-coded metrics).
        * [ ] Enable **report generation** in PDF/Markdown summarizing:

        * Data transformations.
        * Model configuration.
        * Forecast accuracy and diagnostics.

        ---

        #### **8. Version Roadmap**

        * **v1.1** → Custom dataset upload, Plotly visualizations
        * **v1.2** → ARIMA/SARIMA grid search and residual diagnostics
        * **v1.3** → Prophet parameter tuning and enhancements
        * **v1.4** → Auto-Forecast module (model comparison and selection)
        """
    )

# These are future featres i will implement later. Help me rewrite these soi can later implement them

# Need to note in forecast.py that note that (S)ARMA models need stationarity

# in explore.py add a rolling mean plot in explore # - We can also look at the rolling mean and std.

# In transform.py couuld remove seasonal differencing

# In explore.py do residual analysis
# - The residuals should approximate a Gaussian distribution (aka white noise).
# - Visual inspection:
#     - ACF plot.: auto cor in residuals: expect no correlations 
#     - Histogram.
#     - QQ plot. =# # - White noise should ideally follow a normal distribution.
# - Statistical tests:
#     - Normality.
# - &#x2705; Histogram/Density plot.
# - 🤔 QQ-plot
# - &#x274C; Jarque-Bera (reliable for large sample size).
# - &#x274C; Shapiro-Wilk (reliable for large sample size).
#     - Autocorrelation.
#     - Heteroskedasticity.

# grid search for ARIMA and SARIMA with the ability to select
# Filter out non-stationary candidates
# evalauation metrics AIC, BIC, MAPE, RMSE