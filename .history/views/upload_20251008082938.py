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
    """Displays basic info, shape, and head() of the DataFrame."""

    st.divider()
    st.subheader("Data Overview")

    # --- Basic Info ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
    with col2:
        st.write("Data types:")
        st.write(df.dtypes)

    # --- Info Output ---
    with st.expander("See detailed info"):
        buffer = io.StringIO()  # ✅ make sure it's io.StringIO, not list or BytesIO
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        buffer.close()

    # --- Head Preview ---
    st.subheader("Preview Data")
    st.dataframe(df.head(10))
