# explore.py - Enhanced Version
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

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
    dates = df[date_col].tolist()
    
    observed = result.observed.tolist()
    trend = result.trend.tolist()
    seasonal = result.seasonal.tolist()
    residual = result.resid.tolist()
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        subplot_titles=[f"Observed ({target_col})", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=dates, y=observed, mode='lines', name="Observed", line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=trend, mode='lines', name="Trend", line=dict(color='#ff7f0e')), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=seasonal, mode='lines', name="Seasonal", line=dict(color='#2ca02c')), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=residual, mode='lines', name="Residual", line=dict(color='#d62728')), row=4, col=1)
    
    fig.update_layout(
        height=900, 
        title_text=f"Seasonal Decomposition of {target_col}",
        xaxis4_title="Date",
        showlegend=False,
        xaxis=dict(type='date')
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# Enhanced Functions
# -----------------------

def show_comparative_moving_averages(df: pd.DataFrame):
    """Show multiple moving averages for comparison."""
    st.subheader("📊 Moving Average Analysis")
    
    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for moving average.")
        return
    
    # Let users select multiple windows
    st.write("**Compare different smoothing windows:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        window1 = st.number_input("Short-term MA", min_value=2, value=7, step=1)
    with col2:
        window2 = st.number_input("Medium-term MA", min_value=2, value=30, step=1)
    with col3:
        window3 = st.number_input("Long-term MA", min_value=2, value=90, step=1)
    
    # Prepare data
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")
    
    # Calculate moving averages
    df_copy["MA_Short"] = df_copy[target_col].rolling(window=window1).mean()
    df_copy["MA_Medium"] = df_copy[target_col].rolling(window=window2).mean()
    df_copy["MA_Long"] = df_copy[target_col].rolling(window=window3).mean()
    
    dates = df_copy["Date"].tolist()
    original = df_copy[target_col].tolist()
    ma_short = df_copy["MA_Short"].tolist()
    ma_medium = df_copy["MA_Medium"].tolist()
    ma_long = df_copy["MA_Long"].tolist()
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=original, mode="lines", name="Original", 
                            line=dict(color='lightgray', width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=dates, y=ma_short, mode="lines", name=f"{window1}-Period MA",
                            line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=ma_medium, mode="lines", name=f"{window2}-Period MA",
                            line=dict(color='#ff7f0e', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=ma_long, mode="lines", name=f"{window3}-Period MA",
                            line=dict(color='#2ca02c', width=2)))
    
    fig.update_layout(
        title=f"Comparative Moving Averages - {target_col}",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(type="date"),
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    with st.expander("💡 How to interpret"):
        st.write("""
        - **Short-term MA**: Captures recent trends, more responsive to changes
        - **Medium-term MA**: Balances responsiveness and smoothness
        - **Long-term MA**: Shows overall trend, filters out noise
        - **Crossovers**: When short-term crosses above long-term, may indicate upward momentum
        """)


def show_time_series_statistics(df: pd.DataFrame):
    """Show time-series specific statistical analysis."""
    st.subheader("📈 Time Series Statistics")
    
    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column available for analysis.")
        return
    
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")
    
    values = df_copy[target_col].dropna()
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{values.mean():.2f}")
    col2.metric("Std Dev", f"{values.std():.2f}")
    col3.metric("Min", f"{values.min():.2f}")
    col4.metric("Max", f"{values.max():.2f}")
    
    # Stationarity tests
    st.markdown("### 🔍 Stationarity Tests")
    st.write("Tests whether the time series has constant mean and variance over time.")

    with st.expander("💡 Terms"):
        st.write("""
        - [ADF Test](https://rhyslwells.github.io/Data-Archive/categories/data-science/ADF-Test): A test for stationarity
        - [KPSS](https://rhyslwells.github.io/Data-Archive/categories/data-science/KPSS-Test): A test for non-stationarity""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Augmented Dickey-Fuller Test**")
        try:
            adf_result = adfuller(values, autolag='AIC')
            st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
            st.metric("p-value", f"{adf_result[1]:.4f}")
            
            if adf_result[1] < 0.05:
                st.success("✅ Series is likely stationary (p < 0.05)")
            else:
                st.warning("⚠️ Series may be non-stationary (p ≥ 0.05)")
        except Exception as e:
            st.error(f"ADF test failed: {e}")
    
    with col2:
        st.write("**KPSS Test**")
        try:
            kpss_result = kpss(values, regression='c', nlags='auto')
            st.metric("KPSS Statistic", f"{kpss_result[0]:.4f}")
            st.metric("p-value", f"{kpss_result[1]:.4f}")
            
            if kpss_result[1] > 0.05:
                st.success("✅ Series is likely stationary (p > 0.05)")
            else:
                st.warning("⚠️ Series may be non-stationary (p ≤ 0.05)")
        except Exception as e:
            st.error(f"KPSS test failed: {e}")
    
    # Distribution analysis
    st.markdown("### 📊 Distribution Analysis")

    with st.expander("💡 Terms"):
        st.write("""
        - [Distributions](https://rhyslwells.github.io/Data-Archive/categories/statistics/Distributions): The probability distribution of a random variable
        """)
    
    values_list = values.tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values_list, nbinsx=30, name="Distribution"))
    fig.update_layout(
        title=f"Distribution of {target_col}",
        xaxis_title=target_col,
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility over time
    st.markdown("### 📉 Rolling Volatility (Standard Deviation)")
    window = st.slider("Rolling window for volatility", 5, 60, 30)
    
    df_copy["Rolling_Std"] = values.rolling(window=window).std()
    
    dates = df_copy["Date"].tolist()
    rolling_std = df_copy["Rolling_Std"].tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=rolling_std, mode='lines', 
                            fill='tozeroy', name=f"{window}-Period Volatility"))
    fig.update_layout(
        title=f"Rolling Volatility - {target_col}",
        xaxis_title="Date",
        yaxis_title="Standard Deviation",
        xaxis=dict(type='date'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 How to interpret"):
        st.write(f"""
        **Rolling Volatility** measures how much the data varies over a moving {window}-period window.
        
        - **High volatility**: Data points are spread out → more uncertainty, rapid changes
        - **Low volatility**: Data points are clustered together → more stable, predictable
        - **Increasing volatility**: May indicate growing instability or regime change
        - **Volatility spikes**: Often correspond to anomalies, shocks, or structural breaks
        
        Use this to identify:
        - Periods of stability vs. turbulence
        - When risk or uncertainty was highest
        - Whether volatility is increasing or decreasing over time
        """)


def show_autocorrelation_analysis(df: pd.DataFrame):
    """Show autocorrelation and partial autocorrelation."""
    st.subheader("🔗 Autocorrelation Analysis")
    
    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column available for analysis.")
        return
    
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")
    
    values = df_copy[target_col].dropna()
    
    lags = st.slider("Number of lags to display", 10, 100, 40)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Autocorrelation Function (ACF)**")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(values, lags=lags, ax=ax)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"ACF plot failed: {e}")
    
    with col2:
        st.write("**Partial Autocorrelation Function (PACF)**")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(values, lags=lags, ax=ax)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"PACF plot failed: {e}")
    
    with st.expander("💡 How to interpret"):
        st.write("""
        - **[ACF](https://rhyslwells.github.io/Data-Archive/categories/data-science/ACF-Plots)**: Shows correlation between the series and its lagged values
        - **[PACF](https://rhyslwells.github.io/Data-Archive/categories/data-science/PACF-Plots)**: Shows direct correlation after removing indirect effects
        - **Significant lags**: Points outside the confidence interval (blue shaded area)
        - **Use for**: Identifying AR and MA orders for ARIMA modeling
        """)


def show_decomposition(df: pd.DataFrame):
    """Time series decomposition with consistent date handling."""
    st.subheader("🧩 Time Series Decomposition")
    
    target_col = st.session_state.get("selected_col", None)
    if target_col is None:
        st.info("No numeric column stored for decomposition.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
    with col2:
        period = st.number_input("Seasonal period", min_value=2, value=12)
    
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"], errors='coerce')
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")
    
    try:
        result = seasonal_decompose(
            df_copy[target_col], 
            model=model_type, 
            period=period
        )
        make_decomposition_plot(df_copy, result, target_col)
        
        with st.expander("💡 Understanding decomposition"):
            st.write(f"""
            **{model_type.capitalize()} Model**: {"Original = Trend + Seasonal + Residual" if model_type == "additive" else "Original = Trend × Seasonal × Residual"}
            
            - **Trend**: Long-term progression of the series
            - **Seasonal**: Regular pattern that repeats every {period} periods
            - **Residual**: Random variation not explained by trend or seasonality
            """)
            
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
        st.info("Try adjusting the seasonal period or check if you have enough data points.")


# -----------------------
# Main App Function
# -----------------------

def app():
    st.title("📊 Time Series Data Explorer")
    
    df = st.session_state.get("df_transform", None)
    date_col = st.session_state.get("date_col", None)
    if df is None or date_col is None:
        st.warning("⚠️ No dataset or date column found. Please upload a file first.")
        return
    
    # Prepare DataFrame for all analyses
    df_copy = df.copy()
    if date_col not in df_copy.columns:
        df_copy = df_copy.reset_index()
    df_copy["Date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy = df_copy.dropna(subset=["Date"])
    df_copy = df_copy.sort_values(by="Date")
    
    # Store target column
    numeric_cols = df_copy.select_dtypes(include=["float", "int"]).columns.tolist()
    if numeric_cols:
        target_col = numeric_cols[0]
        st.session_state["selected_col"] = target_col
    
    # Enhanced tab structure
    tab1, tab2, tab3 = st.tabs(["📈 Smoothing & Trends", "📊 Statistical Analysis", "🧩 Decomposition"])
    
    with tab1:
        show_comparative_moving_averages(df_copy.copy())
    
    with tab2:
        show_time_series_statistics(df_copy.copy())
        st.divider()
        show_autocorrelation_analysis(df_copy.copy())
    
    with tab3:
        show_decomposition(df_copy.copy())