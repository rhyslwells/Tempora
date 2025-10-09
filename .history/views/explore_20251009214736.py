#explore.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

#============================================================
#Helper Plot Functions
#============================================================
def plot_time_series(df: pd.DataFrame, col: str):
    """Plot a single-column time series with Plotly (robust version)."""
    if df.empty:
        st.info("No data available for plotting.")
        return

    plot_df = df.reset_index()
    date_column_name = plot_df.columns[0]

    plot_df[date_column_name] = pd.to_datetime(plot_df[date_column_name], errors='coerce')

    dates = plot_df[date_column_name].tolist()
    values = plot_df[col].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name=col
    ))

    fig.update_layout(
        title=f"Time Series Preview: {col}",
        xaxis_title="Date",
        yaxis_title=col,
        xaxis=dict(type='date')
    )
    st.plotly_chart(fig, use_container_width=True)


def make_decomposition_plot(index, result, target_col):
    """Helper to create decomposition Plotly figure."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[f"Observed ({target_col})", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.resid, name="Residual"), row=4, col=1)
    fig.update_layout(height=900, title_text=f"Seasonal Decomposition of {target_col}")
    return fig


#============================================================
#Main Functions
#============================================================
def show_overview(df: pd.DataFrame):
    st.subheader("🧾 Dataset Overview")
    st.write("**Data Types:**")
    st.write(df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.markdown("###Missing Values Summary")
    if not missing.empty:
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.info("No missing values detected.")

    st.markdown("###Data Preview")
    st.dataframe(df.head(5))


def show_summary_statistics(df: pd.DataFrame):
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(include="all").T)

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for plotting.")
        return

    selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols, key="summary_col")
    plot_df = df[[selected_col]].dropna().copy()
    plot_df[selected_col] = pd.to_numeric(plot_df[selected_col], errors="coerce")

    #Histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=plot_df[selected_col],
        nbinsx=30,
        marker=dict(color="royalblue", line=dict(width=1, color="white"))
    ))
    hist_fig.update_layout(
        title=f"Histogram of {selected_col}",
        xaxis_title=selected_col,
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    #Boxplot
    box_fig = go.Figure()
    box_fig.add_trace(go.Box(
        y=plot_df[selected_col],
        boxpoints="outliers",
        marker_color="indianred"
    ))
    box_fig.update_layout(
        title=f"Boxplot of {selected_col}",
        yaxis_title=selected_col,
        height=350
    )
    st.plotly_chart(box_fig, use_container_width=True)


def show_time_series_view(df: pd.DataFrame):
    st.subheader("📈 Time Series Visualization")

    date_col = st.session_state.get("date_col", None)
    if date_col is None:
        st.warning("⚠️ No date column stored. Please select a date column first.")
        return None, None

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for time series plotting.")
        return date_col, None

    target_col = st.selectbox("Select a numeric column to plot", numeric_cols, key="ts_col")
    st.session_state["selected_col"] = target_col

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy.set_index(date_col, inplace=True)

    plot_time_series(df_copy, target_col)

    return date_col, target_col


def show_moving_average(df: pd.DataFrame):
    st.subheader("📊 Moving Average Smoothing")

    date_col = st.session_state.get("date_col", None)
    target_col = st.session_state.get("selected_col", None)
    if date_col is None or target_col is None:
        st.info("Please select a numeric column in the time series view first.")
        return

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy.set_index(date_col, inplace=True)

    window = st.slider("Select smoothing window size", 3, 60, 7)
    df_copy["Moving Average"] = df_copy[target_col].rolling(window=window).mean()

    #Plot
    plot_df = df_copy.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df[target_col],
                             mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=plot_df[date_col], y=plot_df["Moving Average"],
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

    date_col = st.session_state.get("date_col", None)
    if date_col is None:
        st.warning("⚠️ No date column stored. Please select a date column first.")
        return

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for decomposition.")
        return

    target_col = st.selectbox("Select a numeric column", numeric_cols, key="dec_col")
    model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
    period = st.number_input("Seasonal period", min_value=2, value=12)

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy.set_index(date_col, inplace=True)

    try:
        result = seasonal_decompose(df_copy[target_col].dropna(), model=model_type, period=period)
        fig = make_decomposition_plot(df_copy.index, result, target_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")


#============================================================
#Main App Function
#============================================================
def app():
    st.title("Data Exploration")

    df = st.session_state.get("df_transform", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a file first.")
        return

    #Tabs
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])


    with tab1:
        #Use the date column stored in session_state
        date_col = st.session_state["date_col"]

        #Make a safe copy and ensure datetime
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        df_copy["Date"] = pd.to_datetime(df_copy[date_col], errors="coerce")

        #Pass this df_copy to the plotting functions
        show_time_series_view(df_copy)
        show_moving_average(df_copy)
    with tab2:
        show_overview(df_copy)
        show_summary_statistics(df_copy)

    with tab3:
        show_decomposition(df_copy)
