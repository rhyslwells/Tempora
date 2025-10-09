from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_correlation_analysis,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)
import streamlit as st


# ============================================================
# --- MAIN APP ENTRYPOINT ---
# ============================================================

def app():
    st.title("Data Exploration")
    st.markdown("""
    ## Introduction
    Explore your uploaded dataset — understand its structure, completeness,
    and patterns before modeling or forecasting.
    """)

    # =====================================================
    # Determine which DataFrame to use
    # Priority:
    # 1. df_transform   → manually transformed data (e.g., after smoothing)
    # 2. df_transformed → output of transform.py
    # 3. raw_df         → direct upload from upload.py
    # =====================================================
    df = None

    if "df_transform" in st.session_state:
        df = st.session_state["df_transform"]
    elif "df_transformed" in st.session_state:
        df = st.session_state["df_transformed"]
        st.session_state["df_transform"] = df  # sync for downstream steps
    elif "raw_df" in st.session_state:
        df = st.session_state["raw_df"]
        st.session_state["df_transform"] = df  # initialize transform layer
    else:
        st.warning("No dataset found. Please upload a file on the 'Upload' page first.")
        return

    # Work off a safe copy
    df = df.copy()

    # =====================================================
    # Tabs
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "Plot","Statistics", 
        "Decomposition"
    ])
    with tab1:
        show_time_series_view(df)
        show_moving_average(df)
    with tab2:
        show_overview(df)
        show_summary_statistics(df)
    with tab3:
        show_decomposition(df)
