import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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


def plot_forecast(train, test, test_forecast_values, future_forecast, model_name, col_name):
    """Create an interactive forecast plot showing train, test, test predictions, and future forecast."""
    fig = go.Figure()
    
    # Training data
    train_reset = train.reset_index()
    date_col = train_reset.columns[0]
    train_dates = pd.to_datetime(train_reset[date_col]).tolist()
    train_values = train_reset[col_name].tolist()
    
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_values,
        mode='lines',
        name='Training Data',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Test data (actual) if available
    if test is not None and len(test) > 0:
        test_reset = test.reset_index()
        test_dates = pd.to_datetime(test_reset[date_col]).tolist()
        test_values = test_reset[col_name].tolist()
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_values,
            mode='lines',
            name='Actual (Test)',
            line=dict(color='#2ca02c', width=2)
        ))
        
        # Test predictions if available - connect to last training point
        if test_forecast_values is not None:
            # Get last training point to connect the forecast
            last_train_date = train_dates[-1]
            last_train_value = train_values[-1]
            
            # Prepend the last training point to create visual continuity
            connected_dates = [last_train_date] + test_dates
            connected_values = [last_train_value] + list(test_forecast_values)
            
            fig.add_trace(go.Scatter(
                x=connected_dates,
                y=connected_values,
                mode='lines',
                name='Test Predictions',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
    
    # Future forecast - connect to last data point
    if future_forecast is not None:
        future_dates = future_forecast.index.tolist()
        future_values = future_forecast.values.tolist()
        
        # Get the last point from full dataset to connect
        if test is not None and len(test) > 0:
            last_actual_date = test_dates[-1]
            last_actual_value = test_values[-1]
        else:
            last_actual_date = train_dates[-1]
            last_actual_value = train_values[-1]
        
        # Prepend the last actual point to create visual continuity
        connected_future_dates = [last_actual_date] + future_dates
        connected_future_values = [last_actual_value] + future_values
        
        fig.add_trace(go.Scatter(
            x=connected_future_dates,
            y=connected_future_values,
            mode='lines',
            name=f'Future Forecast',
            line=dict(color='#d62728', width=2.5)
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

def forecast_exponential_smoothing(train, test, forecast_periods, col_name, full_data, method="Triple"):
    """Perform exponential smoothing forecast."""
    try:
        # Fit model on training data only
        if method == "Single":
            model = SimpleExpSmoothing(train[col_name]).fit()
        elif method == "Double":
            model = Holt(train[col_name]).fit()
        else:  # Triple
            seasonal_period = st.session_state.get("seasonal_period", 12)
            
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
        
        # Calculate metrics on test set if available
        test_forecast_values = None
        if test is not None and len(test) > 0:
            test_forecast_values = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast_values)
        else:
            metrics = (None, None, None, None)
        
        # Now refit on FULL data for future forecast
        if test is not None and len(test) > 0:
            if method == "Single":
                model_full = SimpleExpSmoothing(full_data[col_name]).fit()
            elif method == "Double":
                model_full = Holt(full_data[col_name]).fit()
            else:
                try:
                    model_full = ExponentialSmoothing(
                        full_data[col_name],
                        trend='add',
                        seasonal='mul',
                        seasonal_periods=seasonal_period
                    ).fit()
                except:
                    model_full = ExponentialSmoothing(
                        full_data[col_name],
                        trend='add',
                        seasonal='add',
                        seasonal_periods=seasonal_period
                    ).fit()
        else:
            model_full = model
        
        # Generate future forecast starting after the last data point
        last_date = full_data.index[-1]
        freq = full_data.index.freq or pd.infer_freq(full_data.index) or 'D'
        
        future_forecast = model_full.forecast(steps=forecast_periods)
        future_forecast = pd.Series(future_forecast, index=pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=freq
        ))
        
        return future_forecast, test_forecast_values, metrics, model
    
    except Exception as e:
        st.error(f"Exponential Smoothing failed: {str(e)}")
        return None, None, (None, None, None, None), None


def show_exponential_smoothing():
    """Exponential Smoothing forecasting interface."""
    st.subheader("📈 Exponential Smoothing")
    
    st.write("""
    [Exponential smoothing](https://rhyslwells.github.io/Data-Archive/categories/machine-learning/Exponential-Smoothing) methods weight recent observations more heavily than older ones.
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

def forecast_arima(train, test, forecast_periods, col_name, full_data, order):
    """Perform ARIMA forecast."""
    try:
        # Fit model on training data
        model = ARIMA(train[col_name], order=order).fit()
        
        # Calculate metrics on test set if available
        test_forecast_values = None
        if test is not None and len(test) > 0:
            test_forecast_values = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast_values)
        else:
            metrics = (None, None, None, None)
        
        # Refit on full data for future forecast
        if test is not None and len(test) > 0:
            model_full = ARIMA(full_data[col_name], order=order).fit()
        else:
            model_full = model
        
        # Generate future forecast
        last_date = full_data.index[-1]
        freq = full_data.index.freq or pd.infer_freq(full_data.index) or 'D'
        
        future_forecast = model_full.forecast(steps=forecast_periods)
        future_forecast = pd.Series(future_forecast, index=pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=freq
        ))
        
        return future_forecast, test_forecast_values, metrics, model_full
    
    except Exception as e:
        st.error(f"ARIMA failed: {str(e)}")
        return None, None, (None, None, None, None), None


def show_arima():
    """ARIMA forecasting interface."""
    st.subheader("📊 ARIMA (AutoRegressive Integrated Moving Average)")
    
    st.write("""
    [ARIMA](https://rhyslwells.github.io/Data-Archive/categories/data-science/ARIMA) models capture autocorrelations in the data and are highly flexible.
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

def forecast_sarima(train, test, forecast_periods, col_name, full_data, order, seasonal_order):
    """Perform SARIMA forecast."""
    try:
        # Fit model on training data
        model = SARIMAX(
            train[col_name],
            order=order,
            seasonal_order=seasonal_order
        ).fit(disp=False)
        
        # Calculate metrics on test set if available
        test_forecast_values = None
        if test is not None and len(test) > 0:
            test_forecast_values = model.forecast(steps=len(test))
            metrics = calculate_metrics(test[col_name].values, test_forecast_values)
        else:
            metrics = (None, None, None, None)
        
        # Refit on full data for future forecast
        if test is not None and len(test) > 0:
            model_full = SARIMAX(
                full_data[col_name],
                order=order,
                seasonal_order=seasonal_order
            ).fit(disp=False)
        else:
            model_full = model
        
        # Generate future forecast
        last_date = full_data.index[-1]
        freq = full_data.index.freq or pd.infer_freq(full_data.index) or 'D'
        
        future_forecast = model_full.forecast(steps=forecast_periods)
        future_forecast = pd.Series(future_forecast, index=pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq=freq
        ))
        
        return future_forecast, test_forecast_values, metrics, model_full
    
    except Exception as e:
        st.error(f"SARIMA failed: {str(e)}")
        return None, None, (None, None, None, None), None


def show_sarima():
    """SARIMA forecasting interface."""
    st.subheader("🔄 SARIMA (Seasonal ARIMA)")
    
    st.write("""
    [SARIMA](https://rhyslwells.github.io/Data-Archive/categories/machine-learning/SARIMA) extends ARIMA to explicitly model seasonal patterns.
    Best for data with strong, recurring seasonal components. Suitable for stationary time series (use differencing if needed).
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
def forecast_prophet(
    train, test, forecast_periods, col_name, full_data,
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
):
    """Perform Prophet forecast with user-configurable parameters."""
    try:
        from prophet import Prophet
        from prophet.serialize import model_to_json
        import holidays

        # Prepare training data
        train_prophet = train.reset_index()
        train_prophet.columns = ['ds', 'y']


        # Instantiate model with user parameters
        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        model.fit(train_prophet)

        # Evaluate on test data
        test_forecast_values = None
        if test is not None and len(test) > 0:
            test_future = model.make_future_dataframe(periods=len(test), freq='D')
            test_forecast_df = model.predict(test_future)
            test_forecast_values = test_forecast_df['yhat'].values[-len(test):]
            metrics = calculate_metrics(test[col_name].values, test_forecast_values)
        else:
            metrics = (None, None, None, None)

        # Refit on full data for final forecast
        full_prophet = full_data.reset_index()
        full_prophet.columns = ['ds', 'y']
        model_full = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        model_full.fit(full_prophet)

        # Generate forecast
        future = model_full.make_future_dataframe(periods=forecast_periods, freq='D')
        forecast_df = model_full.predict(future)

        last_date = full_data.index[-1]
        freq = full_data.index.freq or pd.infer_freq(full_data.index) or 'D'
        future_forecast = pd.Series(
            forecast_df['yhat'].values[-forecast_periods:],
            index=pd.date_range(start=last_date + pd.Timedelta(days=1),
                                periods=forecast_periods, freq=freq)
        )

        return future_forecast, test_forecast_values, metrics, model_full

    except ImportError:
        st.error("Prophet not installed. Install with: pip install prophet")
        return None, None, (None, None, None, None), None
    except Exception as e:
        st.error(f"Prophet failed: {str(e)}")
        return None, None, (None, None, None, None), None


def show_prophet():

    """Prophet forecasting interface."""
    # st.subheader("🔮 Prophet (Facebook)")
    
    st.write("""
    [Prophet](https://rhyslwells.github.io/Data-Archive/categories/machine-learning/Prophet) is designed for business forecasting with strong seasonal patterns and holidays.
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





