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

    raw_df = st.session_state["raw_df"].copy()

    # --- Date column & index handling ---
    st.subheader("🗓️ Date Configuration")
    date_col = st.selectbox("Select the date column:", raw_df.columns)
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
    raw_df = raw_df.dropna(subset=[date_col])
    raw_df.set_index(date_col, inplace=True)
    raw_df.sort_index(inplace=True)

    # --- Column selection (single) ---
    st.subheader("📈 Column to Transform")
    selected_col = st.selectbox(
        "Select the column to transform:",
        options=raw_df.columns.tolist(),
        index=0
    )
    if not selected_col:
        st.info("Select a column to proceed.")
        return

    # Make a true copy of the selected column
    df = raw_df[[selected_col]].copy()

    # Convert to numeric safely and avoid SettingWithCopyWarning
    df.loc[:, selected_col] = pd.to_numeric(df[selected_col], errors='coerce')

    # Optional: drop rows where conversion failed (NaNs)
    df = df.dropna(subset=[selected_col])


    # --- Ensure numeric ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if selected_col not in numeric_cols:
        st.warning(f"Column '{selected_col}' is non-numeric and cannot have log/differencing applied.")

    # --- Date range restriction ---
    st.subheader("📆 Date Range")
    min_date, max_date = st.slider(
        "Select analysis period:",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
    )
    df = df.loc[min_date:max_date]

    # --- Resampling & aggregation (optional) ---
    st.subheader("⏳ Frequency & Aggregation (Optional)")
    with st.expander("ℹ️ Info: Resampling / Aggregation"):
        st.markdown("""
        Resample your series to a different frequency if needed:
        - `D` = Daily  
        - `B` = Business Days  
        - `W` = Weekly  
        - `M` = Monthly

        Aggregation method:
        - `mean`, `sum`, `min`, `max`, `first`, `last`
        """)
    freq = st.selectbox("Select resampling frequency:", ["None", "D - Daily", "B - Business Days", "W - Weekly", "M - Monthly"], index=0)
    if freq != "None":
        freq_code = freq.split(" - ")[0]
        agg_method = st.selectbox("Aggregation method:", ["mean", "sum", "min", "max", "first", "last"], index=0)
        df_resampled = df.resample(freq_code).agg(agg_method)
    else:
        df_resampled = df.copy()

    # --- Interpolation ---
    st.subheader("🔧 Interpolation")
    interp_method = st.selectbox("Interpolation method:", ["none", "linear", "time", "nearest"])
    if interp_method != "none":
        df_resampled = df_resampled.interpolate(method=interp_method)

    # --- Log Transform ---
    st.subheader("🔢 Log Transform")
    apply_log = st.checkbox("Apply log transform (log(x))")
    if apply_log and selected_col in numeric_cols:
        df_resampled[selected_col] = np.log(df_resampled[selected_col].replace(0, np.nan))

    # --- Detrend / Difference ---
    st.subheader("📉 Detrending / Differencing")
    detrend_method = st.selectbox(
        "Detrending / Differencing:",
        ["None", "First Difference (df.diff())", "Second Difference (df.diff().diff())", "Remove Linear Trend"]
    )

    # --- ADF Test Function ---
    def adf_test(series):
        series = series.dropna()
        if len(series) < 10:
            return "Too few data points for ADF."
        result = adfuller(series)
        return f"ADF Statistic={result[0]:.3f}, p-value={result[1]:.3f}"

    before_adf = adf_test(df_resampled[selected_col])

    # --- Apply detrending / differencing ---
    if detrend_method == "First Difference (df.diff())":
        df_transformed = df_resampled.diff().dropna()
    elif detrend_method == "Second Difference (df.diff().diff())":
        df_transformed = df_resampled.diff().diff().dropna()
    elif detrend_method == "Remove Linear Trend":
        df_transformed = pd.DataFrame(
            detrend(df_resampled, type='linear'),
            index=df_resampled.index,
            columns=df_resampled.columns
        )
    else:
        df_transformed = df_resampled.copy()

    # FIX: Ensure proper dtypes for Arrow serialization
    df_transformed.index = pd.to_datetime(df_transformed.index)
    df_transformed[selected_col] = pd.to_numeric(df_transformed[selected_col], errors='coerce')
    df_transformed = df_transformed.dropna()
    after_adf = adf_test(df_transformed[selected_col])

    # --- Show ADF Before / After ---
    with st.expander("Compare ADF Before vs After"):
        st.markdown(f"**Before:** {before_adf}  \n**After:** {after_adf}")


    # # --- Preview ---
    # st.subheader("🧾 Transformed Data Preview")
    # st.dataframe(df_transformed.head(10))

    # --- Plotly chart ---
    st.subheader("📊 Time Series Preview")
    if not df_transformed.empty:
        # Reset index to make date a column for Plotly
        plot_df = df_transformed.reset_index()
        date_column_name = plot_df.columns[0]
        
        # Ensure datetime and convert to proper format
        plot_df[date_column_name] = pd.to_datetime(plot_df[date_column_name])
        
        # Convert to list/array to avoid any pandas index issues
        dates = plot_df[date_column_name].tolist()
        values = plot_df[selected_col].tolist()
        
        # Use plotly graph_objects for more control
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name=selected_col
        ))
        
        fig.update_layout(
            title=f"Time Series Preview: {selected_col}",
            xaxis_title="Date",
            yaxis_title=selected_col,
            xaxis=dict(type='date')
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # --- Download CSV ---
    csv = df_transformed.to_csv().encode("utf-8")
    st.download_button(
        label="💾 Download Transformed Data as CSV",
        data=csv,
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    # --- Store transformed dataframe in session state ---
    st.session_state["df_transform"] = df_transformed
    st.success("✅ Transformation complete! You can now explore or forecast this dataset.")
