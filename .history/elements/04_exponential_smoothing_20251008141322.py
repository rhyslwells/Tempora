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
