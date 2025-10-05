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
