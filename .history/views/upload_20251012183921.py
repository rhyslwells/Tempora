import streamlit as st
import pandas as pd
import io
import os

def app():
    st.title("📁 Upload Your Data")
    st.write("Upload a CSV or Excel file to begin your time series analysis.")

    # --- Create Tabs ---
    tab_csv, tab_excel, tab_sample = st.tabs(["📄 CSV Upload", "📊 Excel Upload", "🎯 Sample Data"])

    # --- CSV Upload Tab ---
    with tab_csv:
        st.subheader("Upload a CSV File")
        csv_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")

        if csv_file is not None:
            df = pd.read_csv(csv_file)
            st.success(f"✅ Loaded: {csv_file.name}")
            st.session_state["raw_df"] = df
            show_data_summary(df, csv_file.name)

    # --- Excel Upload Tab ---
    with tab_excel:
        st.subheader("Upload an Excel File")
        excel_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], key="excel_uploader")

        if excel_file is not None:
            excel = pd.ExcelFile(excel_file)
            
            if len(excel.sheet_names) > 1:
                st.info(f"📋 Found {len(excel.sheet_names)} sheets in this file")
                sheet = st.selectbox("Select a sheet to load", excel.sheet_names)
            else:
                sheet = excel.sheet_names[0]
                st.info(f"📋 Loading sheet: {sheet}")

            if sheet:
                df = pd.read_excel(excel, sheet_name=sheet)
                st.success(f"✅ Loaded: {excel_file.name} → {sheet}")
                st.session_state["raw_df"] = df
                show_data_summary(df, f"{excel_file.name} ({sheet})")

    # --- Sample Data Tab ---
    with tab_sample:
        st.subheader("Load Sample Dataset")
        st.write("Try out the app with our example dataset: Google stock prices (daily).")
        
        if st.button("📥 Load Sample Data", type="primary"):
            sample_path = os.path.join("sample_data", "GOOGLE_daily.csv")
            try:
                df = pd.read_csv(sample_path)
                st.success("✅ Loaded: GOOGLE_daily.csv (sample data)")
                st.session_state["raw_df"] = df
                show_data_summary(df, "GOOGLE_daily.csv (sample)")
            except FileNotFoundError:
                st.error(f"❌ Sample file not found at {sample_path}")
                st.info("Please ensure the sample_data folder exists with GOOGLE_daily.csv")


def show_data_summary(df: pd.DataFrame, filename: str):
    """Displays a clean, actionable summary of the uploaded data."""
    
    st.divider()
    
    # --- Quick Stats Header ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📋 Columns", df.shape[1])
    with col3:
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("❓ Missing Data", f"{missing_pct:.1f}%")
    with col4:
        date_cols = df.select_dtypes(include=['object']).apply(pd.to_datetime, errors='coerce').notna().any()
        potential_date_cols = date_cols.sum()
        st.metric("📅 Potential Date Columns", potential_date_cols)
    
    # --- Data Quality Check ---
    st.subheader("🔍 Data Quality Check")
    
    issues = []
    warnings = []
    
    # Check for missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        cols_with_missing = missing[missing > 0].to_dict()
        warnings.append(f"Missing values in {len(cols_with_missing)} column(s)")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"{duplicates} duplicate row(s) detected")
    
    # Check for potential date columns
    if potential_date_cols == 0:
        issues.append("No obvious date column detected - time series analysis requires dates")
    
    # Check minimum data points
    if df.shape[0] < 30:
        warnings.append(f"Only {df.shape[0]} rows - more data recommended for robust analysis")
    
    # Display status
    if len(issues) > 0:
        st.error("⚠️ **Issues Found:**")
        for issue in issues:
            st.write(f"- {issue}")
    
    if len(warnings) > 0:
        st.warning("⚡ **Warnings:**")
        for warning in warnings:
            st.write(f"- {warning}")
    
    if len(issues) == 0 and len(warnings) == 0:
        st.success("✅ Data looks good! Ready for analysis.")
    
    # --- Column Information ---
    st.subheader("📋 Column Information")
    
    # Create a clean column summary
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        
        # Detect if column might be a date
        is_potential_date = False
        if dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='raise')
                is_potential_date = True
            except:
                pass
        
        col_info.append({
            'Column': col,
            'Type': dtype,
            'Missing': missing_count,
            'Unique': unique_count,
            'Potential Date?': '✅' if is_potential_date else ''
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True, hide_index=True)
    
    # --- Data Preview ---
    st.subheader("👀 Data Preview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**First 10 rows of {filename}**")
    with col2:
        preview_rows = st.number_input("Rows to show", min_value=5, max_value=100, value=10, step=5)
    
    st.dataframe(df.head(preview_rows), use_container_width=True)
    
    # --- Quick Actions ---
    st.divider()
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Proceed to Data Preparation", type="primary", use_container_width=True):
            st.info("👉 Navigate to the 'Data Preparation' page in the sidebar to continue")
    
    with col2:
        # Download cleaned data info as CSV
        if st.button("📥 Download Column Info", use_container_width=True):
            csv = col_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{filename}_column_info.csv",
                mime="text/csv"
            )