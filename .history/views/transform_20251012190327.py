import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import detrend
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from plotly.subplots import make_subplots


# ----------------------------- #
# 🔧 Core Functions             #
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


def adf_test(series):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    series = series.dropna()
    if len(series) < 10:
        return None, None, "Insufficient data"
    result = adfuller(series, autolag='AIC')
    return result[0], result[1], "Stationary" if result[1] < 0.05 else "Non-stationary"


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


def apply_log_transform(df, col):
    """Apply log(x) transform."""
    df_copy = df.copy()
    df_copy[col] = np.log(df_copy[col].replace(0, np.nan))
    return df_copy


def apply_detrending(df, method):
    """Apply detrending or differencing."""
    if method == "None":
        return df.copy()
    elif method == "First Difference":
        return df.diff().dropna()
    elif method == "Second Difference":
        return df.diff().diff().dropna()
    elif method == "Linear Detrend":
        return pd.DataFrame(
            detrend(df, type='linear'),
            index=df.index,
            columns=df.columns
        )
    return df.copy()


def plot_comparison(df_before, df_after, col, title="Before vs After Transformation"):
    """Plot before and after comparison."""
    df_before_plot = df_before.reset_index()
    df_after_plot = df_after.reset_index()
    
    date_col_before = df_before_plot.columns[0]
    date_col_after = df_after_plot.columns[0]
    
    dates_before = pd.to_datetime(df_before_plot[date_col_before]).tolist()
    dates_after = pd.to_datetime(df_after_plot[date_col_after]).tolist()
    values_before = df_before_plot[col].tolist()
    values_after = df_after_plot[col].tolist()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Original Data", "Transformed Data"],
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(x=dates_before, y=values_before, mode='lines', name='Original',
                  line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates_after, y=values_after, mode='lines', name='Transformed',
                  line=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", type='date', row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    fig.update_layout(height=600, title_text=title, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_adf_comparison(before_stat, before_pval, before_status, after_stat, after_pval, after_status):
    """Display ADF test comparison with visual indicators."""
    st.subheader("📊 Stationarity Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Before Transformation**")
        if before_stat is not None:
            st.metric("ADF Statistic", f"{before_stat:.4f}")
            st.metric("p-value", f"{before_pval:.4f}")
            if before_status == "Stationary":
                st.success(f"✅ {before_status}")
            else:
                st.warning(f"⚠️ {before_status}")
        else:
            st.info(before_status)
    
    with col2:
        st.markdown("**After Transformation**")
        if after_stat is not None:
            st.metric("ADF Statistic", f"{after_stat:.4f}")
            st.metric("p-value", f"{after_pval:.4f}")
            if after_status == "Stationary":
                st.success(f"✅ {after_status}")
            else:
                st.warning(f"⚠️ {after_status}")
        else:
            st.info(after_status)
    
    with st.expander("💡 How to interpret ADF test"):
        st.write("""
        **Augmented Dickey-Fuller (ADF) Test** checks if your data is stationary.
        
        - **p-value < 0.05**: Data is likely **stationary** ✅
          - Mean and variance are constant over time
          - Safe to use for most forecasting models (ARIMA, etc.)
        
        - **p-value ≥ 0.05**: Data is likely **non-stationary** ⚠️
          - Mean/variance changes over time
          - May need differencing or detrending
        
        - **More negative ADF statistic**: Stronger evidence of stationarity
        
        **Why it matters**: Most time series models assume stationarity. Non-stationary data can lead to spurious results.
        """)


# ----------------------------- #
# 🎨 Main App                   #
# ----------------------------- #

def app():
    st.title("🔄 Data Transformation & Preparation")
    st.write("Transform your raw data into a format ready for time series analysis and forecasting.")

    # --- Check for data ---
    if "raw_df" not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Upload' page.")
        st.info("👉 Navigate to the Upload page using the sidebar")
        return
    
    raw_df = st.session_state["raw_df"].copy()

    # ========================================
    # STEP 1: Basic Setup
    # ========================================
    st.header("1️⃣ Basic Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Select Date Column")
        date_col = st.selectbox("Choose the column containing dates:", raw_df.columns, key="date_select")
        st.session_state["date_col"] = date_col
    
    with col2:
        st.subheader("📊 Select Target Column")
        # Show only after date is selected
        df_temp = prepare_dataframe(raw_df, date_col)
        selected_col = st.selectbox("Choose the column to analyze:", df_temp.columns, key="col_select")
        st.session_state["selected_col"] = selected_col
    
    # Prepare data
    df = prepare_dataframe(raw_df, date_col)
    df = select_numeric_column(df, selected_col)
    
    # Store original for comparison
    df_original = df.copy()
    
    # Date range selector
    st.subheader("📆 Date Range")
    min_date = df.index.min().to_pydatetime()
    max_date = df.index.max().to_pydatetime()
    
    date_range = st.slider(
        "Select the time period for analysis:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    df = df.loc[date_range[0]:date_range[1]].copy()
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points", len(df))
    col2.metric("Start Date", df.index.min().strftime('%Y-%m-%d'))
    col3.metric("End Date", df.index.max().strftime('%Y-%m-%d'))

    st.divider()

    # ========================================
    # STEP 2: Frequency & Missing Data
    # ========================================
    st.header("2️⃣ Frequency & Missing Data Handling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Resampling")
        freq_options = {
            "None": "Keep original frequency",
            "D - Daily": "Aggregate to daily",
            "B - Business Days": "Business days only",
            "W - Weekly": "Aggregate to weekly",
            "MS - Monthly": "Monthly (start of month)",
            "M - Monthly": "Monthly (end of month)",
            "Q - Quarterly": "Aggregate to quarterly",
        }
        freq = st.selectbox("Frequency:", list(freq_options.keys()), 
                           format_func=lambda x: freq_options[x])
        
        if freq != "None":
            agg_method = st.selectbox("Aggregation method:", 
                                     ["mean", "sum", "min", "max", "first", "last"])
            with st.expander("💡 When to use resampling"):
                st.write("""
                - **Daily/Weekly/Monthly**: Reduce noise, smooth irregular data
                - **mean**: Average values (good for prices, temperatures)
                - **sum**: Total values (good for sales, counts)
                - **first/last**: Keep specific observations
                """)
        else:
            agg_method = "mean"
    
    with col2:
        st.subheader("🔧 Interpolation")
        interp_method = st.selectbox("Handle missing values:", 
                                     ["none", "linear", "time", "nearest"])
        
        missing_before = df[selected_col].isna().sum()
        if missing_before > 0:
            st.warning(f"⚠️ {missing_before} missing values detected")
        else:
            st.success("✅ No missing values")
        
        with st.expander("💡 Interpolation methods"):
            st.write("""
            - **none**: Keep missing values as-is
            - **linear**: Straight line between points (most common)
            - **time**: Account for varying time intervals
            - **nearest**: Use closest known value
            """)
    
    # Apply frequency and interpolation
    df = resample_dataframe(df, freq, agg_method)
    df = interpolate_dataframe(df, interp_method)
    
    missing_after = df[selected_col].isna().sum()
    if missing_after > 0:
        st.info(f"ℹ️ {missing_after} missing values remaining after interpolation")

    st.divider()

    # ========================================
    # STEP 3: Transformations
    # ========================================
    st.header("3️⃣ Mathematical Transformations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Log Transform")
        apply_log = st.checkbox("Apply logarithmic transformation")
        
        if apply_log:
            st.info("Good for: Exponential growth, stabilizing variance, handling multiplicative seasonality")
            if (df[selected_col] <= 0).any():
                st.error("❌ Cannot apply log transform: data contains zero or negative values")
                apply_log = False
        
        with st.expander("💡 When to use log transform"):
            st.write("""
            **Use log transform when:**
            - Data shows exponential growth
            - Variance increases with level (heteroscedasticity)
            - Multiplicative seasonality present
            - Need to interpret results as percentage changes
            
            **Don't use when:**
            - Data contains zeros or negative values
            - Additive patterns are present
            - Data is already stationary
            """)
    
    with col2:
        st.subheader("📉 Detrending")
        detrend_options = {
            "None": "No detrending",
            "First Difference": "Remove trend via differencing (y_t - y_{t-1})",
            "Second Difference": "Remove stronger trends (diff of diff)",
            "Linear Detrend": "Remove linear trend component"
        }
        detrend_method = st.selectbox("Detrending method:", 
                                      list(detrend_options.keys()),
                                      format_func=lambda x: detrend_options[x])
        
        with st.expander("💡 When to use detrending"):
            st.write("""
            **First Difference**: Most common, removes linear trends
            - Good for: Stock prices, GDP, sales data
            - Creates: Returns or growth rates
            
            **Second Difference**: Removes quadratic trends
            - Good for: Accelerating growth patterns
            - Use sparingly: loses many observations
            
            **Linear Detrend**: Removes straight-line trend
            - Good for: Data with clear linear component
            - Preserves: Seasonal patterns better than differencing
            
            **None**: When data is already stationary
            """)
    
    # Apply transformations
    df_transformed = df.copy()
    
    if apply_log:
        df_transformed = apply_log_transform(df_transformed, selected_col)
        df_transformed = df_transformed.dropna()
    
    df_transformed = apply_detrending(df_transformed, detrend_method)
    df_transformed = df_transformed.dropna()

    st.divider()

    # ========================================
    # STEP 4: Results & Validation
    # ========================================
    st.header("4️⃣ Transformation Results")
    
    # ADF tests
    before_stat, before_pval, before_status = adf_test(df[selected_col])
    after_stat, after_pval, after_status = adf_test(df_transformed[selected_col])
    
    show_adf_comparison(before_stat, before_pval, before_status, 
                       after_stat, after_pval, after_status)
    
    st.divider()
    
    # Visual comparison
    st.subheader("📊 Visual Comparison")
    plot_comparison(df, df_transformed, selected_col)
    
    # Summary statistics
    with st.expander("📈 Summary Statistics Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Transformation**")
            st.dataframe(df[selected_col].describe())
        
        with col2:
            st.write("**After Transformation**")
            st.dataframe(df_transformed[selected_col].describe())
    
    st.divider()
    
    # ========================================
    # STEP 5: Save & Export
    # ========================================
    st.header("5️⃣ Save & Continue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Store in session state
        if st.button("💾 Save Transformed Data", type="primary", use_container_width=True):
            st.session_state["df_transform"] = df_transformed
            st.success("✅ Transformation saved! You can now proceed to explore or forecast.")
            st.balloons()
    
    with col2:
        # Download option
        csv = df_transformed.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"transformed_{selected_col}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Transformation summary
    with st.expander("📋 Transformation Summary"):
        st.write("**Applied Transformations:**")
        transformations = []
        
        if freq != "None":
            transformations.append(f"- Resampled to {freq} using {agg_method}")
        if interp_method != "none":
            transformations.append(f"- Interpolated missing values using {interp_method}")
        if apply_log:
            transformations.append("- Applied logarithmic transformation")
        if detrend_method != "None":
            transformations.append(f"- Applied {detrend_method}")
        
        if transformations:
            for t in transformations:
                st.write(t)
        else:
            st.write("No transformations applied")
        
        st.write(f"\n**Data points**: {len(df)} → {len(df_transformed)}")
        st.write(f"**Stationarity**: {before_status} → {after_status}")