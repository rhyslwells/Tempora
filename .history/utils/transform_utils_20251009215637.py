# explore.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------
# Helper Plot Functions
# -----------------------

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

    st.plotly_chart(fig, use_container_width=True)


def make_decomposition_plot(df, result, target_col):
    """Create decomposition Plotly figure using robust index handling."""
    plot_df = df.reset_index()
    date_column_name = plot_df.columns[0]

    # Ensure datetime type
    plot_df[date_column_name] = pd.to_datetime(plot_df[date_column_name], errors='coerce')

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[f"Observed ({target_col})", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=900, title_text=f"Seasonal Decomposition of {target_col}",
                      xaxis=dict(type='date'))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Main Functions
# -----------------------

def show_overview(df: pd.DataFrame):
    st.subheader("🧾 Dataset Overview")
    st.write("**Data Types:**")
    st.write(df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.markdown("### Missing Values Summary")
    if not missing.empty:
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.info("No missing values detected.")

    st.markdown("### Data Preview")
    st.dataframe(df.head(5))


def show_summary_statistics(df: pd.DataFrame):
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(include="all").T)


def show_time_series_view(df: pd.DataFrame):
    st.subheader("📈 Time Series Visualization")

    date_col = "Date"
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for time series plotting.")
        return date_col, None

    target_col = numeric_cols[0]  # Only one numeric column assumed
    st.session_state["selected_col"] = target_col

    plot_time_series(df, target_col)
    return date_col, target_col


def show_moving_average(df: pd.DataFrame):
    st.subheader("📊 Moving Average Smoothing")

    date_col = "Date"
    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for moving average.")
        return

    window = st.slider("Select smoothing window size", 3, 60, 7)

    # Compute MA
    df_copy = df.copy()
    df_copy["Moving Average"] = df_copy[target_col].rolling(window=window).mean()

    # Plot with robust time handling
    plot_df = df_copy.reset_index()
    date_column_name = plot_df.columns[0]
    plot_df[date_column_name] = pd.to_datetime(plot_df[date_column_name], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=plot_df[target_col],
                             mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=plot_df[date_column_name], y=plot_df["Moving Average"],
                             mode="lines", name=f"{window}-Period MA"))
    fig.update_layout(
        title=f"Moving Average - {target_col}",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(type="date")
    )
    st.plotly_chart(fig, use_container_width=True)


def show_decomposition(df: pd.DataFrame):
    st.subheader("🧩 Time Series Decomposition")

    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for decomposition.")
        return

    model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
    period = st.number_input("Seasonal period", min_value=2, value=12)

    try:
        result = seasonal_decompose(df[target_col].dropna(), model=model_type, period=period)
        make_decomposition_plot(df, result, target_col)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")


# -----------------------
# Main App Function
# -----------------------

def app():
    st.title("Data Exploration")

    df = st.session_state.get("df_transform", None)
    date_col = st.session_state.get("date_col", None)
    if df is None or date_col is None:
        st.warning("⚠️ No dataset or date column found. Please upload a file first.")
        return

    # Ensure datetime for the stored date column
    df_copy = df.copy()
    df_copy.reset_index(inplace=True)
    df_copy["Date"] = pd.to_datetime(df_copy[date_col], errors="coerce")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        show_time_series_view(df_copy)
        show_moving_average(df_copy)

    with tab2:
        show_overview(df_copy)
        show_summary_statistics(df_copy)

    with tab3:
        show_decomposition(df_copy)
