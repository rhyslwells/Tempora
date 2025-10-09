# explore.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------
# Helper Plot Functions
# -----------------------

def plot_time_series(df, col):
    """Plot a single-column time series with Plotly."""
    if df.empty or col not in df.columns:
        st.info("No data available for plotting.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df[col],
        mode="lines+markers",
        name=col
    ))
    fig.update_layout(
        title=f"Time Series: {col}",
        xaxis_title="Date",
        yaxis_title=col,
        xaxis=dict(type="date")
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_moving_average(df, col, window=7):
    """Plot moving average over time using the same datetime index."""
    if df.empty or col not in df.columns:
        st.info("No data available for plotting moving average.")
        return

    df_copy = df.copy()
    df_copy["MA"] = df_copy[col].rolling(window=window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_copy["Date"], y=df_copy[col], mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=df_copy["Date"], y=df_copy["MA"], mode="lines", name=f"{window}-Period MA"))
    fig.update_layout(
        title=f"Moving Average - {col}",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(type="date")
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_decomposition(df, col, model="additive", period=12):
    """Perform and plot seasonal decomposition."""
    if df.empty or col not in df.columns:
        st.info("No data available for decomposition.")
        return

    result = seasonal_decompose(df[col], model=model, period=period, extrapolate_trend='freq')

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=[f"Observed ({col})", "Trend", "Seasonal", "Residual"]
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(
        height=900,
        title_text=f"Seasonal Decomposition: {col}",
        xaxis=dict(type="date")
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Main Functions
# -----------------------

def show_overview(df):
    st.subheader("🧾 Dataset Overview")
    st.write(df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.markdown("### Missing Values")
    if not missing.empty:
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.info("No missing values detected.")

    st.markdown("### Data Preview")
    st.dataframe(df.head(5))


def show_summary_statistics(df):
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(include="all").T)


# -----------------------
# Main App
# -----------------------

def app():
    st.title("Data Exploration")

    df = st.session_state.get("df_transform", None)
    date_col = st.session_state.get("date_col", None)

    if df is None or date_col is None:
        st.warning("⚠️ No dataset or date column found. Please upload a file first.")
        return

    # --- Prepare a consistent df_copy with Date column ---
    df_copy = df.copy()
    if date_col not in df_copy.columns:
        df_copy = df_copy.reset_index()
    df_copy["Date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")

    # Assume single numeric column
    numeric_cols = df_copy.select_dtypes(include=["float", "int"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for plotting.")
        return
    target_col = numeric_cols[0]
    st.session_state["selected_col"] = target_col

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        plot_time_series(df_copy, target_col)
        window = st.slider("Select moving average window", 3, 60, 7)
        plot_moving_average(df_copy, target_col, window=window)

    with tab2:
        show_overview(df_copy)
        show_summary_statistics(df_copy)

    with tab3:
        model_type = st.radio("Decomposition Model", ["additive", "multiplicative"], horizontal=True)
        period = st.number_input("Seasonal period", min_value=2, value=12)
        plot_decomposition(df_copy, target_col, model=model_type, period=period)
