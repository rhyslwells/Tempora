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
        type_summary = {
            col: list(df[col].map(type).value_counts().index[:3]) for col in df.columns
        }
        st.json(type_summary)


# ============================================================
# --- ARROW SAFETY CLEANER ---
# ============================================================
def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and converts a DataFrame to be safely serializable by Arrow."""
    df = df.copy()

    st.info("🧪 Running Arrow compatibility cleaning...")

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        df.index = df.index.astype(str)

    for col in df.columns:
        try:
            series = df[col]

            # Handle datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                df[col] = pd.to_datetime(series, errors="coerce")
                continue

            # Convert numeric-like objects
            if pd.api.types.is_object_dtype(series):
                # Try numeric conversion first
                converted = pd.to_numeric(series, errors="ignore")
                if pd.api.types.is_numeric_dtype(converted):
                    df[col] = converted
                    continue

            # Fix numpy scalar types (np.float64, np.int64)
            df[col] = series.map(lambda x: x.item() if isinstance(x, np.generic) else x)

            # Convert remaining objects to strings
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

    # =====================================================
    # Load dataset priority
    # =====================================================
    df = None
    if "df_transform" in st.session_state:
        df = st.session_state["df_transform"]
        st.success("✅ Using transformed dataset (`df_transform`).")
    elif "raw_df" in st.session_state:
        df = st.session_state["raw_df"]
        st.info("📄 Using raw uploaded dataset (`raw_df`).")
    else:
        st.warning("⚠️ No dataset found. Please upload a file on the 'Upload' page first.")
        return

    debug_df_info(df, "Before Arrow Compatibility")

    # =====================================================
    # Clean for Arrow / Streamlit serialization
    # =====================================================
    df = make_arrow_compatible(df).copy()

    # =====================================================
    # Ensure datetime index
    # =====================================================
    if not isinstance(df.index, pd.DatetimeIndex):
        datetime_cols = [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()
        ]
        if datetime_cols:
            st.write(f"📅 Using `{datetime_cols[0]}` as datetime index.")
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            df.set_index(datetime_cols[0], inplace=True)
        else:
            st.warning("⚠️ No datetime column found — time series plots may not render correctly.")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    if df.index.name is None:
        df.index.name = "Date"

    debug_df_info(df, "Final Pre-Plot DataFrame")

    # =====================================================
    # Tabs
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        date_col, target_col = show_time_series_view(df)
        if date_col and target_col:
            show_moving_average(df, date_col, target_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)

# we now have:

st.session_state["date_col"] = date_col
