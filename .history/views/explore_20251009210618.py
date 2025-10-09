import streamlit as st
import pandas as pd
import numpy as np
from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)

# ============================================================
# --- UTILITY: Make DataFrame Arrow-Compatible for Streamlit ---
# ============================================================
def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and converts a DataFrame to be safely serializable by Arrow (Streamlit backend)."""
    df = df.copy()

    # --- Ensure clean index ---
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        df.index = df.index.astype(str)

    # --- Fix column data types ---
    for col in df.columns:
        series = df[col]

        # Handle datetime safely
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = pd.to_datetime(series, errors="coerce")
            continue

        # Convert numeric-looking strings to numbers
        if pd.api.types.is_object_dtype(series):
            converted = pd.to_numeric(series, errors="ignore")
            if pd.api.types.is_numeric_dtype(converted):
                df[col] = converted
                continue

        # Fix numpy scalar types (e.g. np.float64)
        df[col] = series.map(lambda x: x.item() if isinstance(x, np.generic) else x)

        # Convert remaining objects to strings
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Replace infinities and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert numeric columns to float64 (Arrow-safe)
    for col in df.select_dtypes(include=["int", "float"]).columns:
        df[col] = df[col].astype("float64")

    return df


# ============================================================
# --- MAIN APP ENTRYPOINT ---
# ============================================================
def app():
    st.title("Data Exploration")

    # =====================================================
    # Load dataset priority:
    # 1. df_transform (user-transformed data)
    # 2. raw_df (uploaded data)
    # =====================================================
    df = None

    if "df_transform" in st.session_state:
        df = st.session_state["df_transform"]
    elif "raw_df" in st.session_state:
        df = st.session_state["raw_df"]
    else:
        st.warning("No dataset found. Please upload a file on the 'Upload' page first.")
        return

    # =====================================================
    # Make DataFrame Arrow-safe and copy for exploration
    # =====================================================
    df = make_arrow_compatible(df).copy()

    # =====================================================
    # Ensure datetime index for time series plotting
    # =====================================================
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to detect a datetime-like column
        datetime_cols = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()
        ]

        if datetime_cols:
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            df.set_index(datetime_cols[0], inplace=True)
        else:
            st.warning("⚠️ No datetime column found — time series plots may not render correctly.")

    # Ensure datetime index validity
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop invalid or NaT indices
    df = df[~df.index.isna()]

    # Assign readable name to index
    if df.index.name is None:
        df.index.name = "Date"

    # =====================================================
    # Streamlit Tabs
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # Let user pick date + target once, then reuse for moving average
        date_col, target_col = show_time_series_view(df)
        if date_col and target_col:
            show_moving_average(df, date_col, target_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)
