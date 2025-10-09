import streamlit as st
import pandas as pd
from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
    show_time_series_view,
    show_moving_average,
    show_decomposition,
)


def app():
    st.title("Data Exploration")

    # --- Load dataset ---
    df = st.session_state.get("df_transform", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a file first.")
        return

    # =====================================================
    # --- Tabs ---
    # =====================================================
    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        # print df columns
        st.write(df.head(10))
        

        ts_date, ts_col = show_time_series_view(df)
        if ts_date and ts_col:
            st.session_state["selected_col"] = ts_col
            show_moving_average(df, ts_date, ts_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)

    with tab3:
        show_decomposition(df)
