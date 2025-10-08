"""
transform.py
------------
Prepare a raw DataFrame for time series analysis.
Allows interactive selection of columns, frequency settings,
interpolation, log transformations, and preview of the transformed data.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Transform Data", layout="wide")
st.title(":orange[Data Transformation]")

st.markdown("""
This module prepares your dataset **before exploration or forecasting**.  
Use it to:
- Select relevant columns  
- Set a date index and frequency  
- Interpolate missing values  
- Apply log transformations  
- Generate a clean DataFrame for downstream analysis
""")

# --- Upload Section ---
st.subheader(":green[1. Upload your dataset]")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully")
    st.write("**Preview of raw data:**")
    st.dataframe(df_raw.head())

    tab1, tab2, tab3 = st.tabs(["Column Selection", "Frequency & Interpolation", "Transformations"])

    # ============================
    # TAB 1: Column Selection
    # ============================
    with tab1:
        st.subheader(":orange[Select Columns]")
        all_cols = list(df_raw.columns)
        selected_cols = st.multiselect("Choose columns to include", all_cols, default=all_cols)
        df_selected = df_raw[selected_cols]
        st.write("**Selected Data:**")
        st.dataframe(df_selected.head())

        st.info("Next → Go to 'Frequency & Interpolation' tab to define the time index and fill missing data.")

    # ============================
    # TAB 2: Frequency & Interpolation
    # ============================
    with tab2:
        st.subheader(":orange[Date Index & Frequency Settings]")

        # Identify potential date columns
        date_cols = [c for c in df_selected.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = st.selectbox("Select date/time column", options=date_cols if date_cols else all_cols)
        st.write(f"Selected Date Column: **{date_col}**")

        df_selected[date_col] = pd.to_datetime(df_selected[date_col], errors="coerce")
        df_selected = df_selected.set_index(date_col).sort_index()

        freq_option = st.selectbox(
            "Set desired frequency for resampling",
            ["Auto-detect", "D (Daily)", "B (Business Day)", "W (Weekly)", "M (Month End)", "Q (Quarter End)", "A (Year End)"],
            index=1
        )

        if freq_option != "Auto-detect":
            freq_map = {
                "D (Daily)": "D",
                "B (Business Day)": "B",
                "W (Weekly)": "W",
                "M (Month End)": "M",
                "Q (Quarter End)": "Q",
                "A (Year End)": "A"
            }
            freq = freq_map.get(freq_option)
            df_resampled = df_selected.resample(freq).mean()
        else:
            df_resampled = df_selected.copy()

        st.markdown("### Interpolation Options")
        interp_method = st.selectbox(
            "Choose interpolation method", 
            ["none", "linear", "time", "spline", "nearest"]
        )

        if interp_method != "none":
            df_resampled = df_resampled.interpolate(method=interp_method)

        st.write("**Preview after resampling/interpolation:**")
        st.dataframe(df_resampled.head())

        st.info("Next → Go to 'Transformations' tab to apply log or other transformations.")

    # ============================
    # TAB 3: Transformations
    # ============================
    with tab3:
        st.subheader(":orange[Apply Transformations]")

        log_transform = st.checkbox("Apply Log10 Transform")
        diff_transform = st.checkbox("Apply First Difference")
        standardize = st.checkbox("Standardize (Z-score)")

        df_transformed = df_resampled.copy()

        if log_transform:
            st.write("Applying log10 transformation...")
            df_transformed = np.log10(df_transformed.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        if diff_transform:
            st.write("Applying first differencing...")
            df_transformed = df_transformed.diff().dropna()

        if standardize:
            st.write("Standardizing columns...")
            df_transformed = (df_transformed - df_transformed.mean()) / df_transformed.std()

        st.markdown("### :green[Transformed Data Preview]")
        st.dataframe(df_transformed.head())

        st.markdown("### Download Transformed Data")
        csv = df_transformed.to_csv().encode('utf-8')
        st.download_button(
            label="Download Transformed CSV",
            data=csv,
            file_name='transformed_data.csv',
            mime='text/csv'
        )

else:
    st.info("👆 Please upload a CSV file to begin transformation.")

