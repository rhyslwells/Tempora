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

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import detrend
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go


# ----------------------------- #
# 🔧 Data Preparation Functions #
# ----------------------------- #

def prepare_dataframe(raw_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Converts date column to datetime and sets it as index."""
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def select_numeric_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ensures selected column is numeric."""
    df = df[[col]].copy()
    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col])
    return df


def restrict_date_range(df: pd.DataFrame, selected_col: str) -> pd.DataFrame:
    """UI date range slider and returns subset of data."""
    st.subheader("📆 Date Range")
    min_date, max_date = st.slider(
        "Select analysis period:",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
    )
    return df.loc[min_date:max_date]


# ----------------------------- #
# 🔁 Transformations            #
# ----------------------------- #

def resample_dataframe(df, freq_option, agg_method):
    """Resample DataFrame to new frequency."""
    if freq_option == "None":
        return df.copy()
    freq_code = freq_option.split(" - ")[0]
    return df.resample(freq_code).agg(agg_method)


def interpolate_dataframe(df, method):
    """Interpolate missing values."""
    if method == "none":
        return df
    return df.interpolate(method=method)


def apply_log_transform(df, col, numeric_cols):
    """Apply log(x) transform to numeric columns."""
    if col in numeric_cols:
        df[col] = np.log(df[col].replace(0, np.nan))
    return df


def apply_detrending(df, method):
    """Apply detrending or differencing."""
    if method == "First Difference (df.diff())":
        return df.diff().dropna()
    elif method == "Second Difference (df.diff().diff())":
        return df.diff().diff().dropna()
    elif method == "Remove Linear Trend":
        return pd.DataFrame(
            detrend(df, type='linear'),
            index=df.index,
            columns=df.columns
        )
    return df.copy()


# ----------------------------- #
# 🧪 Statistical Tests          #
# ----------------------------- #

def adf_test(series):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    series = series.dropna()
    if len(series) < 10:
        return "Too few data points for ADF."
    result = adfuller(series)
    return f"ADF Statistic={result[0]:.3f}, p-value={result[1]:.3f}"


# ----------------------------- #
# 📊 Visualization & Output     #
# ----------------------------- #

def plot_time_series(df, col):
    """Plot a single-column time series with Plotly (robust version)."""
    if df.empty:
        st.info("No data available for plotting.")
        return

    # Reset index to make date a column for Plotly
    plot_df = df.reset_index()
    date_column_name = plot_df.columns[0]

    # Ensure datetime type for the x-axis
    plot_df[date_column_name] = pd.to_datetime(plot_df[date_column_name], errors='coerce')

    # Convert to lists (avoids Arrow serialization issues with certain dtypes)
    dates = plot_df[date_column_name].tolist()
    values = plot_df[col].tolist()

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name=col
    ))

    # Update layout for clarity and consistency
    fig.update_layout(
        title=f"Time Series Preview: {col}",
        xaxis_title="Date",
        yaxis_title=col,
        xaxis=dict(type='date')
    )

    # Render in Streamlit
    st.plotly_chart(fig, use_container_width=True)




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
    st.session_state["date_col"] = date_col
    df = prepare_dataframe(raw_df, date_col).copy()

    # --- Column selection
    st.subheader("📈 Column to Transform")
    selected_col = st.selectbox("Select column:", df.columns)
    st.session_state["selected_col"] = selected_col
    df = select_numeric_column(df, selected_col).copy()

    # --- Date range restriction
    df = restrict_date_range(df, selected_col).copy()

    # --- Resampling / Aggregation
    st.subheader("⏳ Frequency & Aggregation")
    freq = st.selectbox("Frequency:", ["None", "D - Daily", "B - Business Days", "W - Weekly", "M - Monthly"]) # Choose start of the month,week
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
