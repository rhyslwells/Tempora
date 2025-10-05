"""
05_arima_modeling.py
--------------------
Fit and forecast using ARIMA and auto_arima.
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_predict
from 01_data_prep import create_AnalysisData, companyNames

data = create_AnalysisData(companyNames)
company = 'GOOGLE'
rolling = data.rolling(30).mean().dropna()

split = 100
train, test = rolling[company][:-split], rolling[company][-split:]
train_idx, test_idx = rolling.index[:-split], rolling.index[-split:]

auto_model = auto_arima(train)
print(auto_model.summary())

arima_fit = ARIMA(train, order=(1, 1, 0)).fit()
print(arima_fit.summary())

plot_predict(arima_fit, start="2023-08-02", end="2024-01-10")
plt.title(f"ARIMA Predictions - {company}")
plt.show()
