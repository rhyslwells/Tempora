import streamlit as st
import pandas as pd
from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)

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

    # Work off a safe copy
    df = df.copy()

    # =====================================================
    # Ensure datetime index for time series plotting
    # =====================================================
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find a datetime-like column and set it as index
        datetime_cols = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()
        ]

        if datetime_cols:
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            df.set_index(datetime_cols[0], inplace=True)
        else:
            st.warning("⚠️ No datetime column found — time series plots may not render correctly.")

    # Clean up: drop rows with invalid datetimes or NaT index
    df = df[~df.index.isna()]

    # Make sure index has a readable name
    if df.index.name is None:
        df.index.name = "Date"

    # =====================================================
    # Tabs
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # Let the user pick date + target once in time series, then reuse for moving average
        date_col, target_col = show_time_series_view(df)
        if date_col and target_col:
            show_moving_average(df, date_col, target_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)
