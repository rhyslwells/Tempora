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

        * [ ] Add **automated grid search** for ARIMA and SARIMA parameter combinations $(p, d, q)$ and $(P, D, Q, m)$.
        * [ ] **Skip non-stationary models** automatically using the Augmented Dickey–Fuller (ADF) test.
        * [ ] Evaluate all valid models using **performance metrics**:
              * Mean Absolute Error (MAE)
              * Root Mean Squared Error (RMSE)
              * Mean Absolute Percentage Error (MAPE)
              * Akaike Information Criterion (AIC)
              * Bayesian Information Criterion (BIC)
        * [ ] Allow **user selection of preferred metric** for model ranking and comparison.
        * [ ] **Automatically select and fit** the best-performing model based on the chosen metric.
        * [ ] Add **forecast visualization** displaying predicted values with confidence intervals.


        ---

        #### **7. Documentation & Learning**

        * [ ] Add **inline explanations** for each model type and parameter.
        * [ ] Include **“Learn More” expandable sections** beside model results.
        * [ ] Add **progress indicators** (spinners or progress bars) for model fitting and grid search.
        * [ ] Enable **report generation** in PDF/Markdown summarizing:
                * Data transformations.
                * Model configuration.
                * Forecast accuracy and diagnostics.
        ---

        #### **8. Version Roadmap**

        * **v1.1** → Residual analysis in explore, rolling mean in explore, seasonal differencing in transform.
        * **v1.3** → Prophet parameter tuning and enhancements
        * **v1.4** → Auto-Forecast module (model comparison and selection)
        """
    )

