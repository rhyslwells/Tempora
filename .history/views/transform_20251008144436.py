import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import detrend
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

def app():
    st.title("🔄 Data Transformation")
    st.write("""
    Prepare your dataset for time series analysis or forecasting.
    Use the options below to clean, resample, and transform your data.
    """)

    # --- Check for uploaded data ---
    if "raw_df" not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Upload' tab.")
        return

    df = st.session_state["raw_df"].copy()

    # --- Date column & index handling ---
    st.subheader("🗓️ Date Configuration")
    date_col = st.selectbox("Select the date column:", df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # --- Column selection ---
    st.subheader("📈 Columns to Transform")
    selected_cols = st.multiselect(
        "Select columns to include in transformation:",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:1]
    )
    if not selected_cols:
        st.info("Select at least one column to proceed.")
        return

    df = df[selected_cols]

    # --- Numeric column check ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    non_numeric_cols = set(selected_cols) - set(numeric_cols)
    if non_numeric_cols:
        st.warning(f"Columns {non_numeric_cols} are non-numeric and cannot have log/differencing applied.")

    # --- Date range restriction ---
    st.subheader("📆 Date Range")
    min_date, max_date = st.slider(
        "Select analysis period:",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
    )
    df = df.loc[min_date:max_date]

    # --- Frequency resampling & aggregation ---
    st.subheader("⏳ Frequency & Aggregation")
    with st.expander("ℹ️ Info: Resampling"):
        st.markdown("""
        **Frequency Resampling:**  
        - `D` = Daily  
        - `B` = Business Days  
        - `W` = Weekly  
        - `M` = Monthly  

        **Aggregation Method:**  
        - `mean` → average (default)  
        - `sum` → total (useful for counts)  
        - `min` / `max` → minimum or maximum in period  
        - `first` / `last` → first or last value
        """)
    freq = st.selectbox(
        "Select resampling frequency:",
        ["D - Daily", "B - Business Days", "W - Weekly", "M - Monthly"],
        index=1
    )
    freq_code = freq.split(" - ")[0]

    agg_method = st.selectbox(
        "Aggregation method for resampling:",
        ["mean", "sum", "min", "max", "first", "last"],
        index=0
    )
    df = df.resample(freq_code).agg(agg_method)

    # --- Interpolation ---
    st.subheader("🔧 Interpolation")
    with st.expander("ℹ️ Info: Interpolation"):
        st.markdown("""
        Fill missing values after resampling:
        - `none` → leave as NaN  
        - `linear` → linear interpolation  
        - `time` → linear interpolation using time index  
        - `nearest` → fill using nearest observation
        """)
    interp_method = st.selectbox("Interpolation method:", ["none", "linear", "time", "nearest"])
    if interp_method != "none":
        df = df.interpolate(method=interp_method)

    # --- Log transform ---
    st.subheader("🔢 Log Transform")
    with st.expander("ℹ️ Info: Log Transform"):
        st.markdown("""
        Applies log(x) to numeric columns.  
        Useful for:
        - Reducing skewness
        - Stabilizing variance
        - Comparing relative changes
        """)
    apply_log = st.checkbox("Apply log transform (log(x))")
    if apply_log:
        if len(numeric_cols) > 0:
            df[numeric_cols] = np.log(df[numeric_cols].replace(0, np.nan))
        else:
            st.warning("No numeric columns selected for log transform.")

    # --- Detrend / Difference ---
    st.subheader("📉 Detrending / Differencing")
    with st.expander("ℹ️ Info: Detrending / Differencing"):
        st.markdown("""
        Removes trends from your series to make it stationary:
        - `None` → keep original  
        - `First Difference` → x_t - x_{t-1}  
        - `Second Difference` → x_t - 2x_{t-1} + x_{t-2}  
        - `Remove Linear Trend` → removes linear trend using regression
        """)
    detrend_method = st.selectbox(
        "Detrending / Differencing:",
        ["None", "First Difference (df.diff())", "Second Difference (df.diff().diff())", "Remove Linear Trend"]
    )

    # --- Non-stationarity detection ---
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 10:  # avoid very short series
                result = adfuller(series)
                p_value = result[1]
                if p_value > 0.05:
                    st.warning(f"Column '{col}' may be non-stationary (ADF p={p_value:.3f}). Consider differencing/detrending.")

    # --- Apply detrending / differencing ---
    if detrend_method == "First Difference (df.diff())":
        df[numeric_cols] = df[numeric_cols].diff().dropna()
    elif detrend_method == "Second Difference (df.diff().diff())":
        df[numeric_cols] = df[numeric_cols].diff().diff().dropna()
    elif detrend_method == "Remove Linear Trend":
        df[numeric_cols] = df[numeric_cols].apply(lambda x: detrend(x, type='linear'))

    # --- Preview ---
    st.subheader("🧾 Transformed Data Preview")
    st.dataframe(df.head(10))

    # --- Dynamic Plotly chart ---
    st.subheader("📊 Time Series Preview")
    if not df.empty:
        fig = px.line(df, x=df.index, y=df.columns, title="Time Series Preview")
        st.plotly_chart(fig, use_container_width=True)

    # --- Download CSV ---
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="💾 Download Transformed Data as CSV",
        data=csv,
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    # --- Store transformed dataframe in session state ---
    st.session_state["df_transform"] = df
    st.success("✅ Transformation complete! You can now explore or forecast this dataset.")
