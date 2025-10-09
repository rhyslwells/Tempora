import streamlit as st
import pandas as pd
from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)

def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans a DataFrame to be safely serializable by Arrow."""
    import numpy as np

    df = df.copy()

    # Ensure numeric / object consistency
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Try numeric conversion
            converted = pd.to_numeric(df[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(converted):
                df[col] = converted
            else:
                df[col] = df[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            # Force numeric columns to float64
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype("float64")

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ============================================================
# --- MAIN APP ENTRYPOINT ---
# ============================================================
def app():
    st.title("Data Exploration")

    # --- Load dataset ---
    df = st.session_state.get("df_transform", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a file first.")
        return

    # --- Clean for Arrow / Streamlit serialization ---
    df = make_arrow_compatible(df).copy()

    # =====================================================
    # --- Select / confirm datetime column ---
    # =====================================================
    if "date_col" not in st.session_state:
        potential_dates = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()
        ]
        if potential_dates:
            date_col = st.selectbox("Select datetime column", potential_dates)
            st.session_state["date_col"] = date_col
        else:
            st.warning("⚠️ No datetime-like column found. Time series may not render correctly.")
            date_col = None
    else:
        date_col = st.session_state["date_col"]

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
        df.index.name = "Date"

    # =====================================================
    # --- Tabs ---
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        ts_date, ts_col = show_time_series_view(df)
        if ts_date and ts_col:
            st.session_state["selected_col"] = ts_col
            show_moving_average(df, ts_date, ts_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)
