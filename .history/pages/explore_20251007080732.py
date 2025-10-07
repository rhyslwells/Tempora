import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title("Data Exploration")

    st.markdown("""
    ## Introduction
    This section provides an overview and exploratory analysis of your uploaded dataset.
    It helps you understand the structure, completeness, and statistical patterns in your data
    before moving on to modeling or forecasting.
    """)

    # --- Load uploaded DataFrame from session state ---
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No dataset found. Please upload a file on the 'Upload' page first.")
        return

    df = st.session_state.df.copy()

    # --- Tabs for analysis ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Summary Statistics",
        "Correlations",
        "Time Series View"
    ])

    # --------------------------------------------------------------------
    # TAB 1: Overview
    # --------------------------------------------------------------------
    with tab1:
        st.markdown("""
        ### Dataset Overview
        This section provides general information about the dataset, 
        including dimensions, column names, data types, and missing values.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)

        # Missing values summary
        st.markdown("### Missing Values Summary")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.dataframe(missing.rename("Missing Count"))
        else:
            st.info("No missing values detected.")

        # Preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10))

    # --------------------------------------------------------------------
    # TAB 2: Summary Statistics
    # --------------------------------------------------------------------
    with tab2:
        st.markdown("""
        ### Summary Statistics
        Explore descriptive statistics for numerical variables and distributions.
        """)

        st.write("#### Descriptive Statistics")
        st.dataframe(df.describe().T)

        st.write("#### Distributions of Numeric Features")
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for plotting.")

    # --------------------------------------------------------------------
    # TAB 3: Correlations
    # --------------------------------------------------------------------
    with tab3:
        st.markdown("""
        ### Correlation Analysis
        Examine relationships between numerical variables using correlation matrices and heatmaps.
        """)

        if len(df.select_dtypes(include=["float", "int"]).columns) >= 2:
            corr = df.corr(numeric_only=True)
            st.dataframe(corr)

            st.write("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numerical columns to compute correlations.")

    # --------------------------------------------------------------------
    # TAB 4: Time Series View
    # --------------------------------------------------------------------
    with tab4:
        st.markdown("""
        ### Time Series Visualization
        If your dataset contains a datetime column, select it below to view how key variables change over time.
        """)

        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        if not datetime_cols:
            # Try to auto-detect any column that looks like a date
            potential_dates = [c for c in df.columns if "date" in c.lower()]
            if potential_dates:
                try:
                    df[potential_dates[0]] = pd.to_datetime(df[potential_dates[0]], errors="coerce")
                    datetime_cols = [potential_dates[0]]
                except Exception:
                    pass

        if datetime_cols:
            date_col = st.selectbox("Select a datetime column", datetime_cols)
            df = df.sort_values(by=date_col)
            df.set_index(date_col, inplace=True)

            numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select a numeric column to plot", numeric_cols)
                st.line_chart(df[target_col])
            else:
                st.info("No numeric columns available for time series plotting.")
        else:
            st.warning("No datetime columns found in the dataset.")
