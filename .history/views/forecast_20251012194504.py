# in the Exponential smoothing section we would want the output forecast to
# be forcasting beyond the test set, not the training set. You would also expect the plot to show this.
# the train

# the forecast that is downloadable would have the actual data, train+test, and the extended forecast dates with values.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ========================================
# Helper Functions
# ========================================

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove NaN values
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return None, None, None, None
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mae, rmse, mape, r2


def plot_forecast(train, test, forecast, model_name, col_name):
    """Create an interactive forecast plot."""
    train_reset = train.reset_index()
    test_reset = test.reset_index() if test is not None and len(test) > 0 else None
    
    date_col = train_reset.columns[0]
    train_dates = pd.to_datetime(train_reset[date_col]).tolist()
    train_values = train_reset[col_name].tolist()
    
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_values,
        mode='lines',
        name='Training Data',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Test data (if available)
    if test_reset is not None:
        test_dates = pd.to_datetime(test_reset[date_col]).tolist()
        test_values = test_reset[col_name].tolist()
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_values,
            mode='lines',
            name='Actual (Test)',
            line=dict(color='gray', width=2, dash='dash')
        ))
    
    # Forecast
    forecast_dates = forecast.index.tolist()
    forecast_values = forecast.values.tolist()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name=f'{model_name} Forecast',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(type='date'),
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_metrics(metrics, test_available=False):
    """Display forecast accuracy metrics."""
    if test_available and metrics[0] is not None:
        st.subheader("📊 Forecast Accuracy Metrics")
        mae, rmse, mape, r2 = metrics
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("R²", f"{r2:.4f}")
        
        with st.expander("💡 How to interpret metrics"):
            st.write("""
            **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted
            - Lower is better
            - Same units as your data
            
            **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
            - Lower is better
            - More sensitive to outliers than MAE
            
            **MAPE (Mean Absolute Percentage Error)**: Average percentage error
            - Lower is better
            - < 10%: Excellent, 10-20%: Good, 20-50%: Acceptable, >50%: Inaccurate
            
            **R² (R-squared)**: Proportion of variance explained
            - Range: 0 to 1 (higher is better)
            - > 0.9: Excellent, 0.7-0.9: Good, 0.5-0.7: Moderate, < 0.5: Poor
            """)


# ========================================
# Exponential Smoothing
# ========================================

def forecast_exponential_smoothing(train, test, forecast_periods, col_name, method="Triple"):
    """Perform exponential smoothing forecast."""
    try:
        if method == "Single":
            model = SimpleExpSmoothing(train[col_name]).fit()
        elif method == "Double":
            model = Holt(train[col_name]).fit()
        else:  # Triple
            # Determine seasonal periods automatically
            seasonal_period = st.session_state.get("seasonal_period", 12)
            
            # Try multiplicative first, fallback to additive if it fails
            try:
                model = ExponentialSmoothing(
                    train[col_name],
                    trend='add',
                    seasonal='mul',
                    seasonal_periods=seasonal_period
                ).fit()
            except:
                model = ExponentialSmoothing(
                    train[col_name],
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_period
                ).fit()
        
        # Generate forecast
        forecast = model.forecast(steps=forecast_periods)
        forecast = pd.Series(forecast, index=pd.date_range(
            start=train.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=train.index.freq or 'D'
        ))
        
        # Calculate metrics if test data available
        if test is not None and len(test) > 0:
            test_forecast = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast)
        else:
            metrics = (None, None, None, None)
        
        return forecast, metrics, model
    
    except Exception as e:
        st.error(f"Exponential Smoothing failed: {str(e)}")
        return None, (None, None, None, None), None


def show_exponential_smoothing():
    """Exponential Smoothing forecasting interface."""
    st.subheader("📈 Exponential Smoothing")
    
    st.write("""
    Exponential smoothing methods weight recent observations more heavily than older ones.
    Ideal for data with trends and/or seasonal patterns.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Smoothing Method",
            ["Single", "Double", "Triple"],
            help="Single: Level only, Double: Level + Trend, Triple: Level + Trend + Seasonality"
        )
    
    with col2:
        if method == "Triple":
            seasonal_period = st.number_input(
                "Seasonal Period",
                min_value=2,
                value=12,
                help="Number of periods in one seasonal cycle (e.g., 12 for monthly data with yearly seasonality)"
            )
            st.session_state["seasonal_period"] = seasonal_period
    
    with st.expander("💡 When to use each method"):
        st.write("""
        **Single (SES)**: No trend, no seasonality
        - Best for: Flat, stable data with random fluctuations
        - Example: Weekly website visits with no growth
        
        **Double (Holt's)**: Trend, but no seasonality
        - Best for: Data with consistent upward/downward trend
        - Example: Growing sales with no seasonal pattern
        
        **Triple (Holt-Winters)**: Both trend and seasonality
        - Best for: Data with recurring seasonal patterns
        - Example: Retail sales (holiday peaks), temperature (seasonal cycles)
        """)
    
    return method


# ========================================
# ARIMA
# ========================================

def forecast_arima(train, test, forecast_periods, col_name, order):
    """Perform ARIMA forecast."""
    try:
        model = ARIMA(train[col_name], order=order).fit()
        
        # Generate forecast
        forecast = model.forecast(steps=forecast_periods)
        forecast = pd.Series(forecast, index=pd.date_range(
            start=train.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=train.index.freq or 'D'
        ))
        
        # Calculate metrics if test data available
        if test is not None and len(test) > 0:
            test_forecast = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast)
        else:
            metrics = (None, None, None, None)
        
        return forecast, metrics, model
    
    except Exception as e:
        st.error(f"ARIMA failed: {str(e)}")
        return None, (None, None, None, None), None


def show_arima():
    """ARIMA forecasting interface."""
    st.subheader("📊 ARIMA (AutoRegressive Integrated Moving Average)")
    
    st.write("""
    ARIMA models capture autocorrelations in the data and are highly flexible.
    Suitable for stationary time series (use differencing if needed).
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        p = st.number_input(
            "p (AR order)",
            min_value=0,
            max_value=10,
            value=1,
            help="Number of lag observations (autoregressive terms)"
        )
    
    with col2:
        d = st.number_input(
            "d (Differencing)",
            min_value=0,
            max_value=2,
            value=1,
            help="Degree of differencing (0=no differencing, 1=first difference)"
        )
    
    with col3:
        q = st.number_input(
            "q (MA order)",
            min_value=0,
            max_value=10,
            value=1,
            help="Order of moving average"
        )
    
    with st.expander("💡 How to choose ARIMA parameters"):
        st.write("""
        **p (Autoregressive)**: How many past values influence the current value
        - Look at PACF plot (from Explore tab)
        - Start with p=1 or p=2
        
        **d (Differencing)**: Make data stationary
        - d=0: Already stationary
        - d=1: First difference (most common)
        - d=2: Second difference (rarely needed)
        - Check ADF test from Transform tab
        
        **q (Moving Average)**: Dependency on past forecast errors
        - Look at ACF plot (from Explore tab)
        - Start with q=1 or q=2
        
        **Common starting points:**
        - ARIMA(1,1,1): General purpose
        - ARIMA(0,1,1): Random walk with drift
        - ARIMA(1,0,0): Simple AR model
        """)
    
    return (p, d, q)


# ========================================
# SARIMA
# ========================================

def forecast_sarima(train, test, forecast_periods, col_name, order, seasonal_order):
    """Perform SARIMA forecast."""
    try:
        model = SARIMAX(
            train[col_name],
            order=order,
            seasonal_order=seasonal_order
        ).fit(disp=False)
        
        # Generate forecast
        forecast = model.forecast(steps=forecast_periods)
        forecast = pd.Series(forecast, index=pd.date_range(
            start=train.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=train.index.freq or 'D'
        ))
        
        # Calculate metrics if test data available
        if test is not None and len(test) > 0:
            test_forecast = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast)
        else:
            metrics = (None, None, None, None)
        
        return forecast, metrics, model
    
    except Exception as e:
        st.error(f"SARIMA failed: {str(e)}")
        return None, (None, None, None, None), None


def show_sarima():
    """SARIMA forecasting interface."""
    st.subheader("🔄 SARIMA (Seasonal ARIMA)")
    
    st.write("""
    SARIMA extends ARIMA to explicitly model seasonal patterns.
    Best for data with strong, recurring seasonal components.
    """)
    
    st.markdown("**Non-Seasonal Parameters**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        p = st.number_input("p (AR)", 0, 10, 1, key="sarima_p")
    with col2:
        d = st.number_input("d (Diff)", 0, 2, 1, key="sarima_d")
    with col3:
        q = st.number_input("q (MA)", 0, 10, 1, key="sarima_q")
    
    st.markdown("**Seasonal Parameters**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        P = st.number_input("P (Seasonal AR)", 0, 5, 1, key="sarima_P")
    with col2:
        D = st.number_input("D (Seasonal Diff)", 0, 2, 1, key="sarima_D")
    with col3:
        Q = st.number_input("Q (Seasonal MA)", 0, 5, 1, key="sarima_Q")
    with col4:
        m = st.number_input("m (Period)", 2, 365, 12, key="sarima_m",
                           help="Length of seasonal cycle (12 for monthly, 7 for weekly, etc.)")
    
    with st.expander("💡 How to choose SARIMA parameters"):
        st.write("""
        **Non-seasonal (p,d,q)**: Same as ARIMA
        
        **Seasonal (P,D,Q,m)**:
        - **P**: Seasonal autoregressive order (usually 0 or 1)
        - **D**: Seasonal differencing (0 or 1)
        - **Q**: Seasonal moving average order (usually 0 or 1)
        - **m**: Seasonal period
          - 12 for monthly data with yearly seasonality
          - 7 for daily data with weekly seasonality
          - 4 for quarterly data with yearly seasonality
        
        **Common patterns:**
        - SARIMA(1,1,1)(1,1,1,12): Full seasonal model
        - SARIMA(0,1,1)(0,1,1,12): Seasonal random walk
        - Start simple: (1,1,1)(1,0,1,12)
        """)
    
    return (p, d, q), (P, D, Q, m)


# ========================================
# Prophet
# ========================================

def forecast_prophet(train, test, forecast_periods, col_name):
    """Perform Prophet forecast."""
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        train_prophet = train.reset_index()
        train_prophet.columns = ['ds', 'y']
        
        # Initialize and fit model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(train_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods, freq='D')
        forecast_df = model.predict(future)
        
        # Extract forecast
        forecast = pd.Series(
            forecast_df['yhat'].values[-forecast_periods:],
            index=pd.date_range(
                start=train.index[-1] + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq=train.index.freq or 'D'
            )
        )
        
        # Calculate metrics if test data available
        if test is not None and len(test) > 0:
            test_future = model.make_future_dataframe(periods=len(test), freq='D')
            test_forecast_df = model.predict(test_future)
            test_forecast = test_forecast_df['yhat'].values[-len(test):]
            metrics = calculate_metrics(test[col_name].values, test_forecast)
        else:
            metrics = (None, None, None, None)
        
        return forecast, metrics, model
    
    except ImportError:
        st.error("Prophet not installed. Install with: pip install prophet")
        return None, (None, None, None, None), None
    except Exception as e:
        st.error(f"Prophet failed: {str(e)}")
        return None, (None, None, None, None), None


def show_prophet():
    """Prophet forecasting interface."""
    st.subheader("🔮 Prophet (Facebook)")
    
    st.write("""
    Prophet is designed for business forecasting with strong seasonal patterns and holidays.
    Handles missing data and outliers automatically, requires minimal parameter tuning.
    """)
    
    with st.expander("💡 When to use Prophet"):
        st.write("""
        **Best for:**
        - Business metrics (sales, revenue, user growth)
        - Strong seasonal patterns (daily, weekly, yearly)
        - Missing data or outliers
        - Multiple seasonalities
        - Holiday effects
        
        **Advantages:**
        - Easy to use (few parameters)
        - Robust to missing data
        - Automatic seasonality detection
        - Intuitive parameter interpretation
        
        **Not ideal for:**
        - High-frequency financial data
        - Data requiring fine-tuned AR/MA terms
        - Very short time series (< 6 months)
        """)
    
    st.info("Prophet uses default seasonality settings. For advanced tuning, consider using the Prophet library directly.")


# ========================================
# Main App
# ========================================

def app():
    st.title("🔮 Time Series Forecasting")
    st.write("Generate forecasts using various statistical and machine learning models.")
    
    # Check for transformed data
    if "df_transform" not in st.session_state:
        st.warning("⚠️ No transformed data found. Please complete data transformation first.")
        st.info("👉 Navigate to the 'Data Transformation' page to prepare your data")
        return
    
    df = st.session_state["df_transform"].copy()
    col_name = st.session_state.get("selected_col")
    
    if col_name is None or col_name not in df.columns:
        st.error("❌ No valid target column found. Please check your transformation.")
        return
    
    # ========================================
    # Setup Section
    # ========================================
    st.header("1️⃣ Forecast Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Split")
        split_method = st.radio(
            "How to validate forecast?",
            ["Use all data (no validation)", "Train-Test Split"],
            help="Train-Test split lets you evaluate forecast accuracy on held-out data"
        )
        
        if split_method == "Train-Test Split":
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=50,
                value=20,
                help="Percentage of data to hold out for testing"
            )
            split_point = int(len(df) * (1 - test_size/100))
            train = df.iloc[:split_point]
            test = df.iloc[split_point:]
            
            st.info(f"Training: {len(train)} points | Testing: {len(test)} points")
        else:
            train = df
            test = None
            st.info(f"Using all {len(train)} data points for training")
    
    with col2:
        st.subheader("🔭 Forecast Horizon")
        forecast_periods = st.number_input(
            "Number of periods to forecast",
            min_value=1,
            max_value=365,
            value=30,
            help="How many time steps into the future to predict"
        )
        
        st.metric("Current Data Points", len(df))
        st.metric("Forecast Horizon", forecast_periods)
    
    st.divider()
    
    # ========================================
    # Model Tabs
    # ========================================
    st.header("2️⃣ Select Forecasting Model")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Exponential Smoothing",
        "📊 ARIMA",
        "🔄 SARIMA",
        "🔮 Prophet"
    ])
    
    # --- Exponential Smoothing ---
    with tab1:
        method = show_exponential_smoothing()
        
        if st.button("🚀 Run Exponential Smoothing", type="primary", key="run_es"):
            with st.spinner("Fitting model and generating forecast..."):
                forecast, metrics, model = forecast_exponential_smoothing(
                    train, test, forecast_periods, col_name, method
                )
                
                if forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, forecast, f"{method} Exponential Smoothing", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index,
                        'Forecast': forecast.values
                    })
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Forecast",
                        data=csv,
                        file_name=f"forecast_es_{method.lower()}.csv",
                        mime="text/csv"
                    )
    
    # --- ARIMA ---
    with tab2:
        order = show_arima()
        
        if st.button("🚀 Run ARIMA", type="primary", key="run_arima"):
            with st.spinner("Fitting ARIMA model..."):
                forecast, metrics, model = forecast_arima(
                    train, test, forecast_periods, col_name, order
                )
                
                if forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, forecast, f"ARIMA{order}", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Model summary
                    with st.expander("📋 Model Summary"):
                        st.text(str(model.summary()))
                    
                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index,
                        'Forecast': forecast.values
                    })
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Forecast",
                        data=csv,
                        file_name=f"forecast_arima_{order[0]}_{order[1]}_{order[2]}.csv",
                        mime="text/csv"
                    )
    
    # --- SARIMA ---
    with tab3:
        order, seasonal_order = show_sarima()
        
        if st.button("🚀 Run SARIMA", type="primary", key="run_sarima"):
            with st.spinner("Fitting SARIMA model... This may take a minute."):
                forecast, metrics, model = forecast_sarima(
                    train, test, forecast_periods, col_name, order, seasonal_order
                )
                
                if forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, forecast, f"SARIMA{order}x{seasonal_order}", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Model summary
                    with st.expander("📋 Model Summary"):
                        st.text(str(model.summary()))
                    
                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index,
                        'Forecast': forecast.values
                    })
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Forecast",
                        data=csv,
                        file_name=f"forecast_sarima.csv",
                        mime="text/csv"
                    )
    
    # --- Prophet ---
    with tab4:
        show_prophet()
        
        if st.button("🚀 Run Prophet", type="primary", key="run_prophet"):
            with st.spinner("Fitting Prophet model..."):
                forecast, metrics, model = forecast_prophet(
                    train, test, forecast_periods, col_name
                )
                
                if forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, forecast, "Prophet", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'Date': forecast.index,
                        'Forecast': forecast.values
                    })
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Forecast",
                        data=csv,
                        file_name=f"forecast_prophet.csv",
                        mime="text/csv"
                    )