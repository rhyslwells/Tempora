"""
01_data_prep.py
----------------
Load MAANG stock data, standardise date ranges, and prepare for analysis.
"""

import pandas as pd
import numpy as np

companyNames = ['AMAZON', 'APPLE', 'GOOGLE', 'META', 'NETFLIX']
minDate, maxDate = '2019-01-01', '2024-01-10'


def standardiseData(dataFrame, minDate=minDate, maxDate=maxDate):
    new_index = pd.date_range(minDate, maxDate, freq='B')
    dataFrame['Date'] = pd.to_datetime(dataFrame['Date'], yearfirst=True)
    dataFrame.set_index('Date', inplace=True)
    dataFrame = dataFrame[minDate:maxDate]
    newDF = dataFrame.reindex(new_index, method='bfill')
    return newDF[['Close']]


def checkData(df, minDate=minDate, maxDate=maxDate):
    days = len(pd.date_range(minDate, maxDate, freq='B'))
    assert len(df) == days
    assert df['Close'].dtypes == 'float64'
    return None


def create_AnalysisData(companyNames, logData=False):
    listDF = []
    for name in companyNames:
        df = pd.read_csv(f'data/{name}_daily.csv')
        df = standardiseData(df)
        checkData(df)
        listDF.append(df)

    finalDF = pd.concat(listDF, axis=1)
    finalDF.columns = companyNames

    if logData:
        finalDF = finalDF.apply(np.log10)

    return finalDF


if __name__ == "__main__":
    data = create_AnalysisData(companyNames)
    data.plot(title="Closing Values of MAANG Stocks")
