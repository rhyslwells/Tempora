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


    def adf_test(series, col_name):
        """
        Runs ADF test on a series and returns a Markdown-friendly string.
        """
        series = series.dropna()
        if len(series) < 10:
            return f"{col_name}: Too few data points for ADF."
        result = adfuller(series)
        p_value = result[1]
        return f"**{col_name}**: ADF Statistic={result[0]:.3f}, p-value={p_value:.3f}"

    # --- ADF before transform ---
    # st.subheader("📊 Stationarity Check: Before Transformation")
    before_adf = {}
    for col in numeric_cols:
        before_adf[col] = adf_test(df[col], col)
        # st.write(before_adf[col])

    # --- Apply detrending / differencing ---
    if detrend_method == "First Difference (df.diff())":
        df_transformed = df[numeric_cols].diff().dropna()
    elif detrend_method == "Second Difference (df.diff().diff())":
        df_transformed = df[numeric_cols].diff().diff().dropna()
    elif detrend_method == "Remove Linear Trend":
        df_transformed = df[numeric_cols].apply(lambda x: detrend(x, type='linear'))
    else:
        df_transformed = df[numeric_cols]

    # Replace numeric columns in df
    df[numeric_cols] = df_transformed

    # --- ADF after transform ---
    # st.subheader("📊 Stationarity Check: After Transformation")
    after_adf = {}
    for col in numeric_cols:
        after_adf[col] = adf_test(df[col], col)
        # st.write(after_adf[col])

    # --- Optionally: compare before vs after in a single table ---
    compare_adf_df = pd.DataFrame({
        "Before": [before_adf[c] for c in numeric_cols],
        "After": [after_adf[c] for c in numeric_cols]
    }, index=numeric_cols)

    with st.expander("Compare ADF Before vs After"):
        st.table(compare_adf_df)


    # --- Preview ---
    st.subheader("🧾 Transformed Data Preview")
    st.dataframe(df.head(10))

    # --- Dynamic Plotly chart ---
    st.subheader("📊 Time Series Preview")
    plot_df = df.dropna(how='any')
    if not plot_df.empty:
        fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title="Time Series Preview", markers=True)
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
