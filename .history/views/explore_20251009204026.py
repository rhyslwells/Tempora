from utils.explore_utils import (
    show_overview,
    show_summary_statistics,
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

    df = (
        st.session_state.get("df_transform")
        or st.session_state.get("df_transformed")
        or st.session_state.get("raw_df")
    )

    if df is None:
        st.warning("No dataset found. Please upload a file on the 'Upload' page first.")
        return

    df = df.copy()

    tab1, tab2, tab3 = st.tabs(["Plot", "Statistics", "Decomposition"])

    with tab1:
        date_col, target_col = show_time_series_view(df)
        show_moving_average(df, date_col, target_col)

    with tab2:
        show_overview(df)
        show_summary_statistics(df)
        show_correlation_analysis(df)

    with tab3:
        show_decomposition(df)
