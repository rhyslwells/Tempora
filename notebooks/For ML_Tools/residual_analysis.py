from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.statespace.sarimax import SARIMAX

qqplot(df['residuals'], line='45');

# residuals from seasoanl decomp

# residuals from sarima modeling

model = SARIMAX(ARMA_1_1, order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
residuals = model_fit.resid
qqplot(residuals, line='45');
model_fit.plot_diagnostics(figsize=(10, 8));
print(model_fit.summary())

# understand the model.summary()
