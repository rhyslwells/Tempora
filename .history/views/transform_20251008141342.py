import streamlit as st
import pandas as pd
import numpy as np

def app():
    st.title("🔄 Data Transformation")
    st.write("""
    Prepare your dataset for time series analysis or forecasting.
    Use the options below to clean, resample, and transform your data.
    """)

    # --- Check for uploaded data ---
    if "raw_df" not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Upload' section.")
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
        df.columns,
        default=df.columns[:1]
    )
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

    # --- Frequency resampling ---
    st.subheader("⏳ Frequency & Interpolation")
    freq = st.selectbox(
        "Select resampling frequency:",
        ["D - Daily", "B - Business Days", "W - Weekly", "M - Monthly"],
        index=1
    )
    freq_code = freq.split(" - ")[0]
    df = df.resample(freq_code).mean()

    interp_method = st.selectbox("Interpolation method:", ["none", "linear", "time", "nearest"])
    if interp_method != "none":
        df = df.interpolate(method=interp_method)

    # --- Log transform ---
    st.subheader("🔢 Transform Options")
    apply_log = st.checkbox("Apply log transform (log(x))")
    if apply_log:
        df = np.log(df.replace(0, np.nan))

    # --- Detrend / Difference ---
    detrend_method = st.selectbox(
        "Detrending / Differencing:",
        ["None", "First Difference (df.diff())", "Second Difference (df.diff().diff())", "Remove Linear Trend"]
    )

    if detrend_method == "First Difference (df.diff())":
        df = df.diff().dropna()
    elif detrend_method == "Second Difference (df.diff().diff())":
        df = df.diff().diff().dropna()
    elif detrend_method == "Remove Linear Trend":
        from scipy.signal import detrend
        df = df.apply(lambda x: detrend(x, type='linear'), axis=0)
        df = pd.DataFrame(df, index=pd.date_range(start=min_date, periods=len(df), freq=freq_code), columns=selected_cols)

    # --- Preview ---
    st.subheader("🧾 Transformed Data Preview")
    st.dataframe(df.head())

    # --- Download option ---
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="💾 Download Transformed Data as CSV",
        data=csv,
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    # --- Store in session for next steps ---
    st.session_state["transformed_df"] = df
    st.success("✅ Transformation complete! You can now explore or forecast this dataset.")
