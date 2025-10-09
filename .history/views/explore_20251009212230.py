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


def debug_df_info(df: pd.DataFrame, stage: str):
    """Print key dataframe diagnostics to Streamlit."""
    with st.expander(f"🧩 Debug Info – {stage}"):
        st.write("**Shape:**", df.shape)
        st.write("**Index Type:**", type(df.index))
        st.write("**Index Name:**", df.index.name)
        st.write("**Column Types:**")
        st.dataframe(df.dtypes.astype(str))
        st.write("**Sample Data (first 5 rows):**")
        st.dataframe(df.head())
        st.write("**Unique Python types per column:**")
        type_summary = {col: list(df[col].map(type).value_counts().index[:3]) for col in df.columns}
        st.json(type_summary)


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and converts a DataFrame to be safely serializable by Arrow."""
    df = df.copy()
    st.info("🧪 Running Arrow compatibility cleaning...")

    # Index
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        df.index = df.index.astype(str)

    # Columns
    for col in df.columns:
        try:
            series = df[col]

            # Handle datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                df[col] = pd.to_datetime(series, errors="coerce")
                continue

            # Convert numeric-like objects
            if pd.api.types.is_object_dtype(series):
                converted = pd.to_numeric(series, errors="ignore")
                if pd.api.types.is_numeric_dtype(converted):
                    df[col] = converted
                    continue

            # Fix numpy scalar types
            df[col] = series.map(lambda x: x.item() if isinstance(x, np.generic) else x)

            # Remaining objects → string
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str)

        except Exception as e:
            st.error(f"⚠️ Error converting column `{col}`: {e}")

    # Replace infinities / NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Force numeric columns to float64
    for col in df.select_dtypes(include=["int", "float"]).columns:
        try:
            df[col] = df[col].astype("float64")
        except Exception as e:
            st.error(f"⚠️ Failed to cast `{col}` to float64: {e}")

    debug_df_info(df, "After make_arrow_compatible()")
    return df


# ============================================================
# --- MAIN APP ENTRYPOINT ---
# ============================================================
def app():
    st.title("Data Exploration")

    # Load dataset
    df = st.session_state.get("df_transform", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a file first.")
        return

    debug_df_info(df, "Before Arrow Compatibility")

    # Clean for Arrow
    df = make_arrow_compatible(df).copy()

    # =====================================================
    # --- Select / confirm datetime column ---
    # =====================================================
    if "date_col" not in st.session_state:
        # Let user select date column manually
        potential_dates = [
            col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()
        ]
        if not potential_dates:
            st.warning("⚠️ No datetime-like column found. Time series may not render correctly.")
            date_col = None
        else:
            date_col = st.selectbox("Select datetime column", potential_dates)
            st.session_state["date_col"] = date_col
    else:
        date_col = st.session_state["date_col"]

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
        df.index.name = "Date"

    debug_df_info(df, "After Date Column Setup")

    # =====================================================
    # Tabs
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # Show time series & store selected numeric column
        ts_date, ts_col = show_time_series_view(df)
        if ts_date and ts_col:
            st.session_state["selected_col"] = ts_col
            show_moving_average(df, ts_date, ts_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)
