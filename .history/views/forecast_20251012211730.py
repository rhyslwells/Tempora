import streamlit as st
import pandas as pd
import numpy as np

# Import forecast utilities
from utils.forecast_utils import (
    calculate_metrics,
    plot_forecast,
    show_metrics,
    forecast_exponential_smoothing,
    show_exponential_smoothing,
    forecast_arima,
    show_arima,
    forecast_sarima,
    show_sarima,
    forecast_prophet,
    show_prophet
)


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
    
    # --- TAB 1: Exponential Smoothing ---
    with tab1:
        method = show_exponential_smoothing()
        
        if st.button("🚀 Run Exponential Smoothing", type="primary", key="run_es"):
            with st.spinner("Fitting model and generating forecast..."):
                future_forecast, test_forecast_values, metrics, model = forecast_exponential_smoothing(
                    train, test, forecast_periods, col_name, df, method
                )
                
                if future_forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, test_forecast_values, future_forecast, 
                                f"{method} Exponential Smoothing", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Create comprehensive download with actual + forecast
                    download_data = []
                    
                    # Add historical data
                    for idx in df.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': df.loc[idx, col_name],
                            'Forecast': np.nan,
                            'Type': 'Historical'
                        })
                    
                    # Add future forecast
                    for idx in future_forecast.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': np.nan,
                            'Forecast': future_forecast.loc[idx],
                            'Type': 'Future Forecast'
                        })
                    
                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Forecast",
                        data=csv,
                        file_name=f"forecast_es_{method.lower()}_complete.csv",
                        mime="text/csv",
                        help="Includes historical data and future forecasts"
                    )
    
    # --- TAB 2: ARIMA ---
    with tab2:
        order = show_arima()
        
        if st.button("🚀 Run ARIMA", type="primary", key="run_arima"):
            with st.spinner("Fitting ARIMA model..."):
                future_forecast, test_forecast_values, metrics, model = forecast_arima(
                    train, test, forecast_periods, col_name, df, order
                )
                
                if future_forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, test_forecast_values, future_forecast, 
                                f"ARIMA{order}", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Model summary
                    with st.expander("📋 Model Summary"):
                        st.text(str(model.summary()))
                    
                    # Create comprehensive download
                    download_data = []
                    for idx in df.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': df.loc[idx, col_name],
                            'Forecast': np.nan,
                            'Type': 'Historical'
                        })
                    for idx in future_forecast.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': np.nan,
                            'Forecast': future_forecast.loc[idx],
                            'Type': 'Future Forecast'
                        })
                    
                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Forecast",
                        data=csv,
                        file_name=f"forecast_sarima_{order[0]}_{order[1]}_{order[2]}_complete.csv",
                        mime="text/csv"
                    )

    # --- TAB 3: SARIMA ---
    with tab3:
        order, seasonal_order = show_sarima()
        
        if st.button("🚀 Run SARIMA", type="primary", key="run_sarima"):
            with st.spinner("Fitting SARIMA model... This may take a minute."):
                future_forecast, test_forecast_values, metrics, model = forecast_sarima(
                    train, test, forecast_periods, col_name, df, order, seasonal_order
                )
                
                if future_forecast is not None:
                    st.success("✅ Forecast completed!")
                    
                    # Plot forecast results
                    plot_forecast(
                        train, test, test_forecast_values, future_forecast,
                        f"SARIMA{order}x{seasonal_order}", col_name
                    )
                    
                    # Display metrics
                    show_metrics(metrics, test is not None and len(test) > 0)
                    
                    # Model summary
                    with st.expander("📋 Model Summary"):
                        st.text(str(model.summary()))
                    
                    # --- Download full forecast ---
                    download_data = []
                    
                    # Historical data
                    for idx in df.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': df.loc[idx, col_name],
                            'Forecast': np.nan,
                            'Type': 'Historical'
                        })
                    
                    # Future forecast data
                    for idx in future_forecast.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': np.nan,
                            'Forecast': future_forecast.loc[idx],
                            'Type': 'Future Forecast'
                        })
                    
                    # Combine and export
                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Forecast",
                        data=csv,
                        file_name=(
                            f"forecast_sarima_"
                            f"{order[0]}_{order[1]}_{order[2]}_"
                            f"{seasonal_order[0]}_{seasonal_order[1]}_"
                            f"{seasonal_order[2]}_{seasonal_order[3]}_complete.csv"
                        ),
                        mime="text/csv"
                    )

    # --- TAB 4: Prophet ---
    with tab4:
        st.subheader("🔮 Prophet Forecasting")

        # --- Parameter Configuration ---
        with st.expander("⚙️ Prophet Configuration", expanded=True):
            st.write("Adjust Prophet model parameters before running the forecast.")
            
            col1, col2 = st.columns(2)
            with col1:
                daily_seasonality = st.checkbox("Daily Seasonality", value=False)
                weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
                yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
            with col2:
                seasonality_mode = st.selectbox(
                    "Seasonality Mode",
                    options=["additive", "multiplicative"],
                    index=0,
                    help="Multiplicative seasonality is useful when seasonal effects grow with trend."
                )
                changepoint_prior_scale = st.slider(
                    "Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.01,
                    help="Controls flexibility of trend changes. Higher values fit more flexible trends."
                )
                seasonality_prior_scale = st.slider(
                    "Seasonality Prior Scale", 1.0, 50.0, 10.0, 1.0,
                    help="Adjusts how strongly seasonality components are fitted."
                )

            include_holidays = st.checkbox("Include UK Holidays", value=False)
        
        # Display Prophet info
        show_prophet()

        if st.button("🚀 Run Prophet", type="primary", key="run_prophet"):
            with st.spinner("Fitting Prophet model..."):
                future_forecast, test_forecast_values, metrics, model = forecast_prophet(
                    train, test, forecast_periods, col_name, df,
                    daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    include_holidays=include_holidays
                )

                if future_forecast is not None:
                    st.success("✅ Forecast completed!")
                    plot_forecast(train, test, test_forecast_values, future_forecast, "Prophet", col_name)
                    show_metrics(metrics, test is not None and len(test) > 0)


                    # --- Download Forecast Data ---
                    download_data = []
                    for idx in df.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': df.loc[idx, col_name],
                            'Forecast': np.nan,
                            'Type': 'Historical'
                        })
                    for idx in future_forecast.index:
                        download_data.append({
                            'Date': idx,
                            'Actual': np.nan,
                            'Forecast': future_forecast.loc[idx],
                            'Type': 'Future Forecast'
                        })

                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Forecast",
                        data=csv,
                        file_name="forecast_prophet_tuned_complete.csv",
                        mime="text/csv"
                    )
