# explore.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------
# Helper Plot Functions
# -----------------------

def plot_time_series(df, col, date_col="Date"):
    """Plot a single-column time series with Plotly (robust version)."""
    if df.empty:
        st.info("No data available for plotting.")
        return

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    df_copy = df_copy.sort_values(by=date_col)

    dates = df_copy[date_col].tolist()
    values = df_copy[col].tolist()

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

def make_decomposition_plot(df, result, target_col, date_col="Date"):
    """Create decomposition Plotly figure with consistent date handling."""
    # Use the same pattern as plot_time_series
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    df_copy = df_copy.sort_values(by=date_col)
    
    dates = df_copy[date_col].tolist()
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        subplot_titles=[f"Observed ({target_col})", "Trend", "Seasonal", "Residual"]
    )
    
    fig.add_trace(go.Scatter(x=dates, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=result.resid, name="Residual"), row=4, col=1)
    
    fig.update_layout(
        height=900, 
        title_text=f"Seasonal Decomposition of {target_col}",
        xaxis4_title="Date",  # Only label bottom x-axis
        xaxis=dict(type='date')
    )
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
    """Moving average with consistent date handling."""
    st.subheader("📊 Moving Average Smoothing")

    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for moving average.")
        return

    window = st.slider("Select smoothing window size", 3, 60, 7)

    # Follow the plot_time_series pattern exactly
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")

    # Calculate moving average
    df_copy["Moving Average"] = df_copy[target_col].rolling(window=window).mean()

    # Extract as lists for plotting (same as plot_time_series)
    dates = df_copy["Date"].tolist()
    original_values = df_copy[target_col].tolist()
    ma_values = df_copy["Moving Average"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=original_values,
        mode="lines", 
        name="Original"
    ))
    fig.add_trace(go.Scatter(
        x=dates, 
        y=ma_values,
        mode="lines", 
        name=f"{window}-Period MA"
    ))
    
    fig.update_layout(
        title=f"Moving Average - {target_col}",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(type="date")
    )
    st.plotly_chart(fig, use_container_width=True)

def show_decomposition(df: pd.DataFrame):
    """Time series decomposition with consistent date handling."""
    st.subheader("🧩 Time Series Decomposition")

    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for decomposition.")
        return

    model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
    period = st.number_input("Seasonal period", min_value=2, value=12)

    # Prepare data with consistent pattern
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")

    try:
        # Perform decomposition on the cleaned data
        result = seasonal_decompose(
            df_copy[target_col], 
            model=model_type, 
            period=period
        )
        # Pass the cleaned dataframe to the plotting function
        make_decomposition_plot(df_copy, result, target_col)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
        st.info("Try adjusting the seasonal period or check if you have enough data points.")
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

    # Prepare DataFrame for all plots
    df_copy = df.copy()
    if date_col not in df_copy.columns:
        df_copy = df_copy.reset_index()
    df_copy["Date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # We do not need the time series preview, as we havete moving average plot
        # Maybe we allow for multiple different moving averages to be shown.
        df_copy_1 = df_copy.copy()
        show_time_series_view(df_copy_1)
        df_copy_2 = df_copy.copy()
        show_moving_average(df_copy_2)

    with tab2:
        # we do not need dataset overview or missing values or data preview
        # are there an othere statistical things we can explore regarding time series data?
        show_overview(df_copy)
        show_summary_statistics(df_copy)

    with tab3:
        show_decomposition(df_copy)
