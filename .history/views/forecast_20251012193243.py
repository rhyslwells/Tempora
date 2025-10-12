# After we have transform the df to this:

# st.session_state["df_transform"] = df_transformed

# we want to apply expoential smoothing, arima, sarima, prophet on different tabs each on df_transformed

# In forecast.py i want to apply a forecast on the df_transformed for a time period, first i want to select the parameters for each model and then apply the forecast

# use the following as inspiration

# I want this gerneral not applied to stocks

"""
04_exponential_smoothing.py
---------------------------
Single, double, and triple exponential smoothing forecasts.
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from 01_data_prep import create_AnalysisData, companyNames

data = create_AnalysisData(companyNames)
company = 'GOOGLE'
rolling = data.rolling(30).mean().dropna()

split = 100
train, test = rolling[company][:-split], rolling[company][-split:]
train_idx, test_idx = rolling.index[:-split], rolling.index[-split:]

models = {
    "Single": SimpleExpSmoothing(train).fit(),
    "Double": Holt(train).fit(),
    "Triple": ExponentialSmoothing(train, trend='multiplicative',
                                   seasonal='multiplicative', seasonal_periods=13).fit()
}

plt.figure(figsize=(6,4))
plt.plot(train_idx, train, '--b', label='Train')
plt.plot(test_idx, test, '--', color='gray', label='Test')

for name, model in models.items():
    plt.plot(test_idx, model.forecast(len(test)), label=f'{name} Exp Smoothing')

plt.legend(); plt.title(f"Exponential Smoothing - {company}")
plt.show()


"""
05_arima_modeling.py
--------------------
Fit and forecast using ARIMA and auto_arima.
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_predict

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

"""
06_prophet_model.py
-------------------
Forecast using Facebook Prophet.
"""

from prophet import Prophet
import matplotlib.pyplot as plt
from 01_data_prep import create_AnalysisData, companyNames

data = create_AnalysisData(companyNames)
company = 'GOOGLE'

df = data.rolling(30).mean().dropna().reset_index()
df.columns = ['ds'] + companyNames

train = df[['ds', company]].rename(columns={company: 'y'})

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=100, freq='B')
forecast = model.predict(future)

fig = model.plot(forecast)
plt.title(f"{company} Stock Forecast - Prophet")
plt.show()
