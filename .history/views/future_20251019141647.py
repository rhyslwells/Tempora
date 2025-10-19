import streamlit as st

# ---------------- PAGE SETUP ---------------- #
# st.set_page_config(page_title="Future Improvements", layout="wide")

def app():
    st.title(":orange[Future Additions & Improvements]")
    st.markdown(
        "This page outlines planned enhancements for upcoming versions of the **Time Series Analysis App**."
    )

    st.divider()

    # ---------------- MAIN CONTENT ---------------- #
    st.markdown(
        """

        ### **General Improvements**
        - [ ] Add example usage gif to readme.
        - [ ] Add tests for all modules.

        ### **Explore Module**
        Planned improvements to exploratory analysis functionality.

        - Add **residual analysis**
            - [ ] Include histogram and kernel density plots.
            - [ ] Add **normality tests** (Jarque–Bera, Shapiro–Wilk).
            - [ ] Implement **residual analysis suite**:
                - Histogram and QQ plot of residuals.
                - ACF plot of residuals (independence check).
                - Statistical tests for heteroskedasticity and autocorrelation.
        - [ ] Add toggle for **data normalization or scaling** before visualization.
        - [ ] Implement **anomaly detection** using IQR, z-score, or rolling MAD.

        ### **Forecast Module**
        Planned improvements to forecasting functionality.

        - [ ] Add **interactive forecast plots** with shaded confidence intervals.
        - [ ] Add an **“Undo Transform”** button after forecasting to invert applied transformations before download.

        #### **Auto-Forecast Module**
        Planned improvements to auto-forecast functionality within the forecast module.

        - [ ] Add **automated grid search** for ARIMA and SARIMA parameter combinations:
              $(p, d, q)$ and $(P, D, Q, m)$ with user-adjustable sliders.
        - [ ] Automatically **skip non-stationary models** using the Augmented Dickey–Fuller (ADF) test.
        - [ ] Evaluate all valid models using multiple **performance metrics**:
                - Mean Absolute Error (MAE)
                - Root Mean Squared Error (RMSE)
                - Mean Absolute Percentage Error (MAPE)
                - Akaike Information Criterion (AIC)
                - Bayesian Information Criterion (BIC)
        - [ ] Allow user to **select preferred metric** for model ranking and comparison.
        - [ ] **Automatically select and fit** the best-performing model based on the chosen metric.
        - [ ] Add **forecast visualizations** showing predicted values with confidence intervals.
        - [ ] Add **progress indicators** (spinners or progress bars) for model fitting and grid search.

        """
    )
