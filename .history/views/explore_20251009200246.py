from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_correlation_analysis,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)
import streamlit as st
import pandas as pd


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

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No dataset found. Please upload a file on the 'Upload' page first.")
        return

    df = st.session_state.df.copy()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Summary Statistics", "Correlations",
        "Time Series View", "Moving Average", "Decomposition"
    ])

    with tab1:
        show_overview(df)
    with tab2:
        show_summary_statistics(df)
    with tab3:
        show_correlation_analysis(df)
    with tab4:
        show_time_series_view(df)
    with tab5:
        show_moving_average(df)
    with tab6:
        show_decomposition(df)
