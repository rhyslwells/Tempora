"""
02_exploration.py
-----------------
Basic time series visualisation, monthly resampling, and rolling means.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import month_plot
from 01_data_prep import create_AnalysisData, companyNames

timeSeriesData = create_AnalysisData(companyNames)

# Rolling means
rolling = timeSeriesData.rolling(30).mean().dropna()
rolling.plot(title='30-Day Rolling Mean')

# Monthly aggregation
monthly = timeSeriesData.resample('M').sum()
monthly.plot(title='Monthly Aggregated Values')

# Month plot
company = 'GOOGLE'
month_plot(monthly[company][:-1])
plt.title(f"Monthly Seasonal Plot - {company}")
plt.show()

"""
03_decomposition_stats.py
-------------------------
Decompose the series, plot ACF/PACF, and check for stationarity.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss
from 01_data_prep import create_AnalysisData, companyNames

data = create_AnalysisData(companyNames)
company = 'GOOGLE'

# ACF / PACF
plot_acf(data[company])
plot_pacf(data[company])

# Decomposition
decomp = seasonal_decompose(data[company], model='multiplicative', period=30)

def plot_decompose(ts, dec):
    fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(ts); ax[0].set_title('Original')
    ax[1].plot(dec.trend); ax[1].set_title('Trend')
    ax[2].plot(dec.seasonal); ax[2].set_title('Seasonality')
    ax[3].plot(dec.resid); ax[3].set_title('Residuals')
    plt.tight_layout()
    plt.show()

plot_decompose(data[company], decomp)
sns.displot(decomp.resid.dropna())

# Stationarity check
stat, p, *_ = kpss(data[company].dropna())
print(f"KPSS p-value: {p} -> {'Stationary' if p>0.05 else 'Not Stationary'}")

def check_stationarity(train,diff=0):

    statistic, p_value, n_lags, critical_values = kpss(train)

    stationarity_message = f"""

    :red[DIFF = {diff}]

    :blue[KPSS Statistic] : {np.round(statistic,4)}, 

    :blue[p-value]        : {p_value}, 
    
    :blue[num lage]       : {n_lags}, 


    :blue[Critical Values] - 
    {critical_values}

    :orange[RESULT : The series is {"not " if p_value < 0.05 else ""}stationary]

    

                         """

    st.write(stationarity_message)
    
    if p_value < 0.05:
        diff += 1
        train = train.diff().dropna()
        check_stationarity(train,diff=diff)


    return None


def show_monthly_sale(timeSeriesData):

    timeSeriesData_monthly = timeSeriesData.resample('M').sum()
    fig1 = plt.figure(figsize=(12,4))
    ax = fig1.add_subplot(111)
    _ = month_plot(timeSeriesData_monthly[:-1],ax=ax)
    st.pyplot(fig1)

    return None 
