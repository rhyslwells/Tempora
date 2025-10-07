import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Summary Statistics",
        "Correlations",
        "Time Series View",
        "Moving Average",
        "Decomposition"
    ])

    # --------------------------------------------------------------------
    # TAB 1: Overview
    # --------------------------------------------------------------------
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        st.markdown("### Missing Values Summary")
        if not missing.empty:
            st.dataframe(missing.rename("Missing Count"))
        else:
            st.info("No missing values detected.")

        st.markdown("### Data Preview")
        st.dataframe(df.head(10))

    # --------------------------------------------------------------------
    # TAB 2: Summary Statistics
    # --------------------------------------------------------------------
    with tab2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T)

        numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
            fig = px.histogram(df, x=selected_col, marginal="box", nbins=30, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for plotting.")

    # --------------------------------------------------------------------
    # TAB 3: Correlations
    # --------------------------------------------------------------------
    with tab3:
        st.subheader("Correlation Analysis")
        num_cols = df.select_dtypes(include=["float", "int"])
        if len(num_cols.columns) >= 2:
            corr = num_cols.corr()
            st.dataframe(corr)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical columns to compute correlations.")

    # --------------------------------------------------------------------
    # TAB 4: Time Series View
    # --------------------------------------------------------------------
    with tab4:
        st.subheader("Time Series Visualization")

        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # Attempt auto-detection
        if not datetime_cols:
            potential_dates = [c for c in df.columns if "date" in c.lower()]
            for c in potential_dates:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    if df[c].notna().any():
                        datetime_cols.append(c)
                except Exception:
                    pass

        if datetime_cols:
            date_col = st.selectbox("Select a datetime column", datetime_cols)
            df = df.sort_values(by=date_col)
            df.set_index(date_col, inplace=True)

            numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select a numeric column to plot", numeric_cols)
                fig = px.line(df, x=df.index, y=target_col, title=f"{target_col} over time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for time series plotting.")
        else:
            st.warning("No datetime columns found in the dataset.")

    # --------------------------------------------------------------------
    # TAB 5: Moving Average (Smoothing)
    # --------------------------------------------------------------------
    with tab5:
        st.subheader("Moving Average Smoothing")

        if not datetime_cols:
            st.warning("No datetime column found. Please check your dataset.")
        else:
            date_col = st.selectbox("Select a datetime column", datetime_cols, key="ma_date")
            df = df.sort_values(by=date_col)
            df.set_index(date_col, inplace=True)

            numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select a numeric column", numeric_cols, key="ma_col")
                window = st.slider("Select smoothing window size", 3, 60, 7)

                df["Moving Average"] = df[target_col].rolling(window=window).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode="lines", name="Original"))
                fig.add_trace(go.Scatter(x=df.index, y=df["Moving Average"], mode="lines", name=f"{window}-Period MA"))
                fig.update_layout(title=f"Moving Average Smoothing - {target_col}", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for smoothing.")

    # --------------------------------------------------------------------
    # TAB 6: Time Series Decomposition
    # --------------------------------------------------------------------
    with tab6:
        st.subheader("Time Series Decomposition")

        if not datetime_cols:
            st.warning("No datetime column found. Please check your dataset.")
        else:
            date_col = st.selectbox("Select a datetime column", datetime_cols, key="dec_date")
            df = df.sort_values(by=date_col)
            df.set_index(date_col, inplace=True)

            numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select a numeric column", numeric_cols, key="dec_col")
                model_type = st.radio("Select model type", ["additive", "multiplicative"], horizontal=True)
                period = st.number_input("Seasonal period", min_value=2, value=12)

                try:
                    result = seasonal_decompose(df[target_col].dropna(), model=model_type, period=period)

                    # Plot decomposition components interactively
                    fig = make_decomposition_plot(df.index, result, target_col)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Decomposition failed: {e}")
            else:
                st.info("No numeric columns available for decomposition.")


def make_decomposition_plot(index, result, target_col):
    """Helper function to create Plotly decomposition figure."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[
        f"Observed ({target_col})", "Trend", "Seasonal", "Residual"
    ])

    fig.add_trace(go.Scatter(x=index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=index, y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=900, title_text=f"Seasonal Decomposition of {target_col}")
    return fig
