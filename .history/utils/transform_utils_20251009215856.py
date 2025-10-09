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

