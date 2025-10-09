import streamlit as st
import pandas as pd
from utils.transform_utils import (
    prepare_dataframe,
    resample_dataframe,
    interpolate_dataframe,
    adf_test,
    plot_time_series,
    apply_detrending,
    apply_log_transform,
    restrict_date_range,
    select_numeric_column,
)

def app():
    st.title("🔄 Data Transformation")
    st.write("Prepare your dataset for time series analysis or forecasting.")

    # --- Load uploaded data
    if "raw_df" not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Upload' tab.")
        return
    raw_df = st.session_state["raw_df"].copy()

    # --- Date setup
    st.subheader("🗓️ Date Configuration")
    date_col = st.selectbox("Select the date column:", raw_df.columns)
    df = prepare_dataframe(raw_df, date_col).copy()

    # --- Column selection
    st.subheader("📈 Column to Transform")
    selected_col = st.selectbox("Select column:", df.columns)
    df = select_numeric_column(df, selected_col).copy()

    # --- Date range restriction
    df = restrict_date_range(df, selected_col).copy()

    # --- Resampling / Aggregation
    st.subheader("⏳ Frequency & Aggregation")
    freq = st.selectbox("Frequency:", ["None", "D - Daily", "B - Business Days", "W - Weekly", "M - Monthly"])
    agg_method = st.selectbox("Aggregation method:", ["mean", "sum", "min", "max", "first", "last"])
    df = resample_dataframe(df, freq, agg_method).copy()

    # --- Interpolation
    interp_method = st.selectbox("Interpolation method:", ["none", "linear", "time", "nearest"])
    df = interpolate_dataframe(df, interp_method).copy()

    # --- Log Transform
    apply_log = st.checkbox("Apply log transform (log(x))")
    if apply_log:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df = apply_log_transform(df, selected_col, numeric_cols).copy()

    # --- Detrending / Differencing
    detrend_method = st.selectbox("Detrending method:", [
        "None",
        "First Difference (df.diff())",
        "Second Difference (df.diff().diff())",
        "Remove Linear Trend"
    ])

    # --- Run ADF before detrending
    before_adf = adf_test(df[selected_col])

    # --- Apply detrending / differencing
    df_transformed = apply_detrending(df.copy(), detrend_method)

    # --- Ensure numeric and Arrow-compatible
    df_transformed[selected_col] = pd.to_numeric(df_transformed[selected_col], errors='coerce')
    df_transformed = df_transformed.dropna().copy()

    # Convert any object columns (esp. index) to string to avoid Arrow serialization errors
    for c in df_transformed.columns:
        if df_transformed[c].dtype == "object":
            df_transformed[c] = df_transformed[c].astype(str)

    # --- ADF after detrending
    after_adf = adf_test(df_transformed[selected_col])

    # --- Compare ADF
    with st.expander("Compare ADF Before vs After"):
        st.markdown(f"**Before:** {before_adf}  \n**After:** {after_adf}")

    # --- Plot
    st.subheader("📊 Time Series Preview")
    plot_time_series(df_transformed, selected_col)

    # --- Download
    csv = df_transformed.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="💾 Download Transformed Data",
        data=csv,
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    # --- Store result
    st.session_state["df_transform"] = df_transformed
    st.success("✅ Transformation complete! You can now explore or forecast this dataset.")
