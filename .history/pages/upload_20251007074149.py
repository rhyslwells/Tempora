# Reformat/edit so that it uploads csv or xlsx (with sheet selection) on different tabs. I only want to focus on these.

# and then give information abotu the loaded dataframe, infor and head()

import streamlit as st
import sqlite3
import datetime
import pandas as pd
import os
import tempfile

from utils.visualization import generate_mermaid_er
import streamlit_mermaid
from utils.db_utils import create_connection

def get_timestamped_db_name():
    return f"db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def app():
    st.title("Upload and Query SQLite DB in Memory")

    # --- Initialization ---
    if "conn" not in st.session_state:
        st.session_state.conn = None
    if "sample_loaded" not in st.session_state:
        st.session_state.sample_loaded = False

    # --- Load Sample DB Button ---
    st.subheader("Try a Sample Database")
    st.markdown("""
    The sample database contains the following:
    - Multiple related tables
    - Example data to test ER diagrams and queries
    - File: `longlist.db` from [CS50’s Introduction to Databases with SQL](https://cs50.harvard.edu/sql/2024/)
    """)

    if st.button("Load Example Database"):
        example_path = os.path.join("sample_data", "longlist.db")
        if os.path.exists(example_path):
            with open(example_path, "rb") as f:
                db_bytes = f.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                tmp_file.write(db_bytes)
                tmp_filename = tmp_file.name

            disk_conn = sqlite3.connect(tmp_filename, check_same_thread=False)
            mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            disk_conn.backup(mem_conn)
            disk_conn.close()
            os.remove(tmp_filename)

            st.session_state.conn = mem_conn
            st.session_state.sample_loaded = True
            st.success("Example database loaded into memory.")
        else:
            st.error("Sample database not found.")

from elements.scr import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA as arima_model


import streamlit as st 
import datetime

######### initial setup 

companyNames = ['AMAZON', 'APPLE', 'GOOGLE', 'META', 'NETFLIX']
st.set_page_config(page_title='TSA@streamlit', layout='wide')
st.title(':orange[Time Series Analysis]')


######### About the app ###############
markdown_about_msg = """
        
        ## Introduction

        The app implements the fundamental steps in time series analysis and forecasting. 
        It lets you play around with key paramters and visualise their effect in the statistics and prediction. Currently the app
        is using a model data set (source below)

        Data source :  KAGGLE : [link](https://www.kaggle.com/datasets/nikhil1e9/netflix-stock-price) to the data set 
        Companies used in the dataset MAANG = Meta, Apple, Amazon, Netflix and, Google

        
        :blue[KeyWords] :  **Time Series Analysis, ARMA, SeriesDecomposition, Forecasting**

    """



############ SIDEBAR for setting the parameters ##########################
with st.sidebar:
    st.header(':red[Chose your Parameters]')
    st.write(" :violet[Running on Streamlit version] -- " + st.__version__)


    # dates = st.date_input(label=':orange[Enter the date range for the analysis]',
    #                       value = (datetime.date(2019,1,1), datetime.date(2024,1,10)),
    #                       min_value=, 
    #                       max_value=datetime.date(2024,1,10),
    #                       format="YYYY-MM-DD")
    
    minDate = st.date_input(label='Enter :orange[minimum] date for analysis', value=datetime.date(2019,1,1),
                             min_value=datetime.date(2018,1,1),
                             max_value=datetime.date(2023,1,1),format="YYYY-MM-DD")
    
    maxDate = st.date_input(label='Enter :orange[maximum] date for analysis', value=datetime.date(2024,1,10),
                             min_value=datetime.date(2022,1,1),
                             max_value=datetime.date(2024,1,10))
    
    if minDate > maxDate:

        st.warning('Minimum Date should be earlier than maximum Date')
    
    # minDate,maxDate = str(dates[0]), str(dates[1])
    logData = st.radio(':orange[Logged Values]', options = [True, False],index=None)
    company = st.radio(':orange[Chose the company to analyse]', options=companyNames)
    window_size = st.slider(':orange[Chose the rolling window size]',min_value=5,max_value=50,value=28)
    monthly_plot = st.button('Show Monthly Plot')


    st.subheader('Lags for ACF and PACF plots')

    lags = st.slider("Set the Lag", min_value=5,max_value=100,value=50)

    ts_decompose_model = st.radio('# :orange[Choose Decomposition Model]', options=['additive', 'multiplicative'])

    split_at = st.number_input(':orange[Split data into Train-Test starting from the end]', 
                               min_value=100,max_value=500,value=250,step=50)
    
    ## ARIMA model parameters
    st.subheader(':green[ARIMA model parameters]')
    p = st.number_input(':green[p]', min_value=0,max_value=30,value=1)
    d = st.number_input(':green[d]', min_value=0,max_value=30,value=1)
    q = st.number_input(':green[q]', min_value=0,max_value=30,value=0)
    


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About", "Exploration", "ARIMA models",
                                              "Prophet"])


with tab1:

    st.markdown(markdown_about_msg)

    def load_data(logData):
        df = create_AnalysisData(companyNames,minDate,maxDate,logData)
        return df

    DF = load_data(logData)

    col1,col2 = st.columns(2,gap="medium")

    with col1:
        st.subheader(':orange[Data frame head]')
        st.dataframe(DF.head(7))

    with col2:

        st.subheader(f':orange[Time Series View]',)
        st.line_chart(DF)

    with st.expander(':orange[Expand to see the summary statistics]'):

        st.write((DF.describe()))


    st.divider()

    ##### End Message
    st.markdown(':orange[:heart: and 🕊️ for all - Chakresh]')

with tab2:

    st.subheader(f'Analysis for :orange[{company}] time series Data: ')

    timeSeriesData = DF[company]
    timeSeriesData_rolling = timeSeriesData.rolling(window_size).mean().dropna()
    st.line_chart(timeSeriesData_rolling)


    if monthly_plot:
        st.subheader(f'Monthly price plot for :orange[{company}] data')
        show_monthly_sale(timeSeriesData)

    st.divider()


    st.subheader('Time Series Decomposition')

    st.write(f'with decomposition model as :blue[{ts_decompose_model}]')

    ts_decomposition = seasonal_decompose(x=timeSeriesData[:-1],model=ts_decompose_model,period=30)

    T,S,R = ts_decomposition.trend, ts_decomposition.seasonal, ts_decomposition.resid

    with st.expander("See the Trend, Seasonality and Residual Plots"):

        st.subheader('Trend')
        st.line_chart(T)
        st.subheader('Seasonality')
        st.line_chart(S)
        st.subheader('Residual')
        st.line_chart(R,width=1)


    # st.header()
        
    st.subheader('Exponential Smoothing - (Done on the rolling average series)')


    smoothing_type = st.multiselect(':orange[Chose Smoothing Type]',options=['Single', 'Double', 'Triple'], default=['Single'])

    # Creating the training and the test data. We do it outside the functions for increased scope. 
    training_Data,test_Data = timeSeriesData_rolling[:-split_at], timeSeriesData_rolling[-split_at:]


    gen_smooth = st.button('Generate Smoothing ')

    if gen_smooth:
        generate_smoothing(smoothing_type,ts_decompose_model,training_Data,test_Data,timeSeriesData_rolling,split_at)

    
    st.subheader('Autoregression Plots')

    show_arplots = st.button("Show ACF and PACF plots")

    if show_arplots:
    
        autoregression_plots(timeSeriesData_rolling,lags=lags)
    
    
    
    st.divider()
