import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import detrend

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
        default=df.columns.tolist()[:1]  # pre-select first column
    )

    if not selected_cols:
        st.info("Select at least one column to proceed.")
        return

    df = df[selected_cols]

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
    st.markdown("""
    **Frequency Resampling:**  
    Choose how to aggregate your data over time:
    - `D` = Daily  
    - `B` = Business Days (excludes weekends/holidays)  
    - `W` = Weekly  
    - `M` = Monthly  

    **Aggregation Method:**  
    When reducing frequency (e.g., daily → weekly), choose how to combine the values:
    - `mean` → average over the period (default for numeric data)  
    - `sum` → total over the period (e.g., sales, volumes)  
    - `min` / `max` → minimum or maximum value in the period  
    - `first` / `last` → use first or last value of the period
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
    st.markdown("""
    After resampling, missing values may appear. Choose a method to fill them:
    - `none` → leave missing values as NaN  
    - `linear` → linear interpolation between points  
    - `time` → linear interpolation using the time index  
    - `nearest` → fill using the nearest valid observation
    """)
    interp_method = st.selectbox("Interpolation method:", ["none", "linear", "time", "nearest"])
    if interp_method != "none":
        df = df.interpolate(method=interp_method)

    # --- Log transform ---
    st.subheader("🔢 Log Transform")
    st.markdown("""
    Applies `log(x)` to the selected columns. Useful for:
    - Reducing skewness
    - Stabilizing variance
    - Comparing relative changes instead of absolute
    """)
    apply_log = st.checkbox("Apply log transform (log(x))")
    if apply_log:
        df = np.log(df.replace(0, np.nan))

    # --- Detrend / Difference ---
    st.subheader("📉 Detrending / Differencing")
    st.markdown("""
    Removes trends from your time series to make it stationary:
    - `None` → keep original series  
    - `First Difference` → \( x_t - x_{t-1} \)  
    - `Second Difference` → \( x_t - 2x_{t-1} + x_{t-2} \)  
    - `Remove Linear Trend` → removes linear trend using regression/detrend
    """)
    detrend_method = st.selectbox(
        "Detrending / Differencing:",
        ["None", "First Difference (df.diff())", "Second Difference (df.diff().diff())", "Remove Linear Trend"]
    )

    if detrend_method == "First Difference (df.diff())":
        df = df.diff().dropna()
    elif detrend_method == "Second Difference (df.diff().diff())":
        df = df.diff().diff().dropna()
    elif detrend_method == "Remove Linear Trend":
        df = df.apply(lambda x: detrend(x, type='linear'), axis=0)
        df = pd.DataFrame(df, index=pd.date_range(start=min_date, periods=len(df), freq=freq_code), columns=selected_cols)

    # --- Preview ---
    st.subheader("🧾 Transformed Data Preview")
    st.dataframe(df.head(10))

    # --- Download option ---
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="💾 Download Transformed Data as CSV",
        data=csv,
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    # --- Store transformed dataframe for downstream pages ---
    st.session_state["df_transform"] = df
    st.success("✅ Transformation complete! You can now explore or forecast this dataset.")
