import streamlit as st
import pandas as pd
from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)


def app():
    st.title("Data Exploration")

    # --- Load dataset ---
    df = st.session_state.get("df_transform", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a file first.")
        return

    # =====================================================
    # --- Tabs ---
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # print df columns
        st.write(df.columns.tolist())
        date_col=st.session_state["date_col"]
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")



        show_time_series_view(df)

        show_moving_average(df)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)

#lets just keep all functions together at the moment

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose


def detect_datetime_columns(df: pd.DataFrame):
    """Try to detect datetime columns automatically."""
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not datetime_cols:
        potential_dates = [c for c in df.columns if "date" in c.lower()]
        for c in potential_dates:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                if df[c].notna().any():
                    datetime_cols.append(c)
            except Exception:
                pass
    return datetime_cols



def show_time_series_view(df: pd.DataFrame):
    """Plot numeric column against datetime index (base visualization)."""
    st.subheader("📈 Time Series Visualization")

    datetime_cols = detect_datetime_columns(df)
    if not datetime_cols:
        st.warning("No datetime columns found in the dataset.")
        return None, None

    date_col = st.selectbox("Select a datetime column", datetime_cols, key="ts_date")
    df = df.sort_values(by=date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for time series plotting.")
        return date_col, None

    target_col = st.selectbox("Select a numeric column to plot", numeric_cols, key="ts_col")

    # --- Plot base time series ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[target_col],
        mode="lines+markers",
        name=target_col
    ))
    fig.update_layout(
        title=f"{target_col} over Time",
        xaxis_title="Date",
        yaxis_title=target_col
    )
    st.plotly_chart(fig, use_container_width=True)

    return date_col, target_col


def show_moving_average(df: pd.DataFrame):
    """Apply and plot moving average smoothing."""
    if not date_col or not target_col:
        st.info("Please select columns in the time series view above first.")
        return

    st.subheader("📊 Moving Average Smoothing")

    df = df.sort_values(by=date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    window = st.slider("Select smoothing window size", 3, 60, 7)
    df["Moving Average"] = df[target_col].rolling(window=window).mean()

    # Store for downstream usage
    st.session_state["df_transform"] = df

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[target_col],
                             mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=df[date_col], y=df["Moving Average"],
                             mode="lines", name=f"{window}-Period MA"))
    fig.update_layout(
        title=f"Moving Average Smoothing - {target_col}",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_overview(df: pd.DataFrame):
    """Display basic dataset info and missing values."""
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
    """Show summary statistics with separate histogram and boxplot."""
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(include="all").T)

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for plotting.")
        return

    selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
    plot_df = df[[selected_col]].dropna().copy()
    plot_df[selected_col] = pd.to_numeric(plot_df[selected_col], errors="coerce")

    # --- Histogram ---
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

    # --- Boxplot ---
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



def make_decomposition_plot(index, result, target_col):
    """Helper to create Plotly decomposition plot."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[f"Observed ({target_col})", "Trend", "Seasonal", "Residual"])

    fig.add_trace(go.Scatter(x=index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.resid, name="Residual"), row=4, col=1)
    fig.update_layout(height=900, title_text=f"Seasonal Decomposition of {target_col}")
    return fig


def show_decomposition(df: pd.DataFrame):
    """Perform and visualize seasonal decomposition."""
    st.subheader("🧩 Time Series Decomposition")

    datetime_cols = detect_datetime_columns(df)
    if not datetime_cols:
        st.warning("No datetime column found.")
        return

    date_col = st.selectbox("Select a datetime column", datetime_cols, key="dec_date")
    df = df.sort_values(by=date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for decomposition.")
        return

    target_col = st.selectbox("Select a numeric column", numeric_cols, key="dec_col")
    model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
    period = st.number_input("Seasonal period", min_value=2, value=12)

    try:
        result = seasonal_decompose(df[target_col].dropna(), model=model_type, period=period)
        fig = make_decomposition_plot(df[date_col], result, target_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
