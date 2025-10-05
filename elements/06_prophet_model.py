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
