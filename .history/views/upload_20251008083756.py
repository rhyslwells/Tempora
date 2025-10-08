# AttributeError: 'list' object has no attribute 'write'
#   colums is shown as a list fix show_data_summary

import streamlit as st
import pandas as pd
import io

def app():
    st.title("Upload CSV or Excel Data")
    st.write("Upload your dataset and explore its structure, columns, and sample data.")

    # --- Create Tabs ---
    tab_csv, tab_excel = st.tabs(["CSV Upload", "Excel Upload"])

    # --- CSV Upload Tab ---
    with tab_csv:
        st.subheader("Upload a CSV File")
        csv_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            st.success("File successfully loaded.")
            show_data_summary(df)

    # --- Excel Uploads Tab ---
    with tab_excel:
        st.subheader("Upload an Excel File")
        excel_file = st.file_uploader("Choose an Excel file", type=["xlsx"], key="excel_uploader")

        if excel_file is not None:
            excel = pd.ExcelFile(excel_file)
            st.write("Sheets found:", excel.sheet_names)

            sheet = st.selectbox("Select a sheet to load", excel.sheet_names)

            if sheet:
                df = pd.read_excel(excel, sheet_name=sheet)
                st.success(f"Loaded sheet: {sheet}")
                show_data_summary(df)


def show_data_summary(df: pd.DataFrame):
    """Displays an enhanced, interactive summary of the DataFrame."""

    st.divider()
    st.subheader("Data Overview")

    # --- Shape and Columns ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df.shape)
        # st.write("**Columns:**", list(df.columns))
    with col2:
        st.write("**Data types:**")
        st.write(df.dtypes)

    # --- Missing Values ---
    missing = df.isna().sum()
    if missing.sum() > 0:
        with st.expander("Missing Values"):
            st.write(missing[missing > 0])
    else:
        st.success("No missing values detected!")

    # # --- Detailed df.info() ---
    # with st.expander("Detailed Info (df.info)"):
    #     with io.StringIO() as buffer:
    #         df.info(buf=buffer)
    #         info_str = buffer.getvalue()
    #         st.text(info_str)

    # --- Unique Values for Categorical Columns ---
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        with st.expander("Unique Values (Categorical Columns)"):
            for col in categorical_cols:
                st.write(f"**{col}:** {df[col].nunique()} unique values")

    # --- Column Selector ---
    st.subheader("Select Columns to Explore")
    selected_columns = st.multiselect(
        "Choose columns to inspect:",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:5]  # pre-select first 5 columns
    )

    if selected_columns:
        selected_df = df[selected_columns]

        # --- Basic Statistics (Numeric) ---
        numeric_cols = selected_df.select_dtypes(include=["int", "float"]).columns
        if len(numeric_cols) > 0:
            with st.expander("Basic Statistics (Numeric Columns)"):
                st.dataframe(selected_df[numeric_cols].describe().T)

        # --- Head Preview ---
        st.subheader("Preview Data")
        st.dataframe(selected_df.head(10))
    else:
        st.info("Select one or more columns above to explore data.")
