# %% [markdown]
# # ARMA, ARIMA, SARIMA
# 
# <img src="media/cover.png" style="width: 40%; display: block; margin: auto;">

# %% [markdown]
# ## Overview
# 
# - In the previous chapters, we covered stationarity, smoothing, trend, seasonality, and autocorrelation, and built two different models: 
# 
# > **MA models**: The current value of the series depends linearly on the series' mean and a set of prior (observed) white noise error terms.
# 
# > **AR models**: The current value of the  series depends linearly on its own previous values and on a stochastic term (an imperfectly predictable term).
# 
# - In this chapter we will review these concepts and combine the AR and MA models into three more complicated ones.

# %% [markdown]
# In particular, we will cover:
# 
# 1. Autoregressive Moving Average (ARMA) models.
# 2. Autoregressive Integrated Moving Average (ARIMA) models.
# 3. SARIMA models (ARIMA model for data with seasonality).
# 4. Selecting the best model.

# %%
import sys

# Install dependencies if the notebook is running in Colab
if 'google.colab' in sys.modules:
    !pip install -U -qq tsa-course pmdarima numpy==1.26

# %%
# Imports
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', UserWarning)
import time
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as ss
import seaborn as sns
from tqdm.notebook import tqdm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import mse
from statsmodels.tsa.statespace.tools import diff
import pmdarima as pm
from tsa_course.lecture1 import fft_analysis
np.random.seed(0)                

# %% [markdown]
# ---

# %% [markdown]
# ## ARMA
# 
# The ARMA model (also known as the *Box-Jenkins* approach) combines two models:
# 
# - An autoregressive (AR) model of order $p$.
# - A moving average (MA) model of order $q$.

# %% [markdown]
# - When we have autocorrelation between outcomes and their ancestors, there will be a pattern in the time series. 
# - This relationship can be modeled using an ARMA model. 
# - It allows us to predict the future with a confidence level proportional to the strength of the relationship and the proximity to known values (prediction weakens the further out we go).

# %% [markdown]
# ```{note}
# - ARMA models assume the time series is stationary.
# - A good rule of thumb is to have at least 100 observations when fitting an ARMA model.
# ```

# %% [markdown]
# ### Example data
# 
# - In the following, we'll look at the monthly average temperatures between 1907-1972.

# %%
# load data and convert to datetime
monthly_temp = pd.read_csv('https://zenodo.org/records/10951538/files/arima_temp.csv?download=1', 
                           skipfooter=2, 
                           header=0, 
                           index_col=0, 
                           names=['month', 'temp'],
                           engine='python')
monthly_temp.index = pd.to_datetime(monthly_temp.index)

# %% [markdown]
# - This is how the data looks like.

# %%
monthly_temp.head()

# %% [markdown]
# - These are some statistics.

# %%
monthly_temp.describe()

# %% [markdown]
# - This is the run sequence plot.

# %%
monthly_temp['temp'].plot(grid=True, figsize=(14, 4), title="Monthly temperatures");

# %% [markdown]
# - We compute the annual mean and plot it on top of the data.

# %%
# Compute annual mean 
annual_temp = monthly_temp.resample('YE').mean()
annual_temp.index.name = 'year'

plt.figure(figsize=(14, 4))
plt.plot(monthly_temp, label="Monthly Temperatures")
plt.plot(annual_temp, label="Annual Mean")
plt.grid(); plt.legend();

# %% [markdown]
# - This gives us an indication that the mean is rather constant over the years.
# - We can extract further information abouth the underlying trend and seasonality by performing a seasonal decomposition.
# - We can use both the `seasonal_decompose` and the `STL` methods.

# %%
decomposition = seasonal_decompose(x=monthly_temp['temp'], model='additive', period=12)
seasonal, trend, resid = decomposition.seasonal, decomposition.trend, decomposition.resid

fig, axs = plt.subplots(2,2, sharex=True, figsize=(18,6))
axs[0,0].plot(monthly_temp['temp'])
axs[0,0].set_title('Original')
axs[0,1].plot(seasonal)
axs[0,1].set_title('Seasonal')
axs[1,0].plot(trend)
axs[1,0].set_title('Trend')
axs[1,1].plot(resid)
axs[1,1].set_title('Residual')
plt.tight_layout()

# %%
decomposition = STL(endog=monthly_temp['temp'], period=12, seasonal=13, robust=True).fit()
seasonal, trend, resid = decomposition.seasonal, decomposition.trend, decomposition.resid

fig, axs = plt.subplots(2,2, sharex=True, figsize=(18,6))
axs[0,0].plot(monthly_temp['temp'])
axs[0,0].set_title('Original')
axs[0,1].plot(seasonal)
axs[0,1].set_title('Seasonal')
axs[1,0].plot(trend)
axs[1,0].set_title('Trend')
axs[1,1].plot(resid)
axs[1,1].set_title('Residual')
plt.tight_layout()

# %% [markdown]
# - The seasonality is well defined.
# - There doesn't seem to be a strong, time-varying trend in the data.
#     - We can assume the trend is almost constant.
#     
# ---

# %% [markdown]
# ## ARMA modeling stages

# %% [markdown]
# There are three stages in building an ARMA model:
# 
# 1. Model identification.
# 2. Model estimation.
# 3. Model evaluation.

# %% [markdown]
# ### Model identification
# 
# - Model identification consists in finding the orders $p$ and $q$ of AR and MA components.
# - Before performing model identification we need to:
#     1. Determine if the time series is stationary.
#     2. Determine if the time series has seasonal component.

# %% [markdown]
# #### Determine stationarity
# 
# - We will use tools we already know (ADF test).
# - We can also look at the rolling mean and std.

# %% [markdown]
# ```{attention}
# Before we continue, let's consider the result below.
# ```

# %%
sinusoid = np.sin(np.arange(200))
_, pvalue, _, _, _, _ = adfuller(sinusoid)
print(f'p-value: {pvalue}')

# %% [markdown]
# - Periodic signals, by their nature, have means and variances that repeat over the period of the cycle. 
# - This implies that their statistical properties are functions of time within each period. 
# - For instance, the mean of a periodic signal over one cycle may be constant.
# - However, when considering any point in time relative to the cycle, the instantaneous mean of the signal can vary. 
# - Similarly, the variance can fluctuate within the cycle.

# %% [markdown]
# - The ADF test specifically looks for a *unit root* (more on this later on).
# - A unit root indicates that shocks to the time series have a permanent effect, causing drifts in the level of the series. 
# - A sinusoidal function, by contrast, is inherently *mean-reverting* within its cycles.
# - After a peak a sinusoid reverts to its mean and any "shock" in terms of phase shift or amplitude change does not alter its oscillatory nature.

# %% [markdown]
# - It's crucial to note that the ADF test's conclusion of stationarity for a sinusoid does not imply that the sinusoid is stationary. 
# - The test's conclusion is about the *absence of a unit root*.
# - This does not imply that the mean and variance are constant within the periodic fluctuations.

# %%
def adftest(series, plots=True):
    out = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {out[0]:.2f}')
    print(f'p-value: {out[1]:.3f}')
    print(f"Critical Values: {[f'{k}: {r:.2f}' for r,k in zip(out[4].values(), out[4].keys())]}\n")
    
    if plots:
        # Compute rolling statistics
        rolmean = series.rolling(window=12).mean()
        rolstd = series.rolling(window=12).std()

        # Plot rolling statistics:
        plt.figure(figsize=(14, 4))
        plt.plot(series, color='tab:blue',label='Original')
        plt.plot(rolmean, color='tab:red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.grid(); 

# %%
# run ADF on monthly temperatures
adftest(monthly_temp.temp)

# %%
# run ADF on annual means
adftest(annual_temp.temp, plots=False)

# %% [markdown]
# - The $p$-value indicates that the time series is stationary...
# - ... even if it clearly has a periodic component.
# - The rolling mean and rolling standard deviation seem globally constant along the time series...
# - ... even if they change locally within the period.

# %% [markdown]
# #### Determine seasonality
# 
# We can determine if seasonality is present by using the following tools:
# - Autocorrelation plot.
# - Seasonal subseries plot (month plot).
# - Fourier Transform.

# %% [markdown]
# Let's first look how these plots look like on synthetic data

# %%
# Generate synthetic time series data
dates = pd.date_range(start='2010-01-01', periods=60, freq='M')  # Monthly data for 5 years
seas = 12 # change this and see how the plots change
data = (np.sin(np.arange(60)*2*np.pi/seas) + 
        np.random.normal(loc=0, scale=0.2, size=60))  # Seasonal data with noise
series = pd.Series(data, index=dates)

# %%
fig, axes = plt.subplots(1,3,figsize=(16,4))
series.plot(ax=axes[0], title="Original time series")

# ACF Plot
plot_acf(series, lags=36, ax=axes[1]);

# Convert series to a DataFrame and add a column for the month
df = series.to_frame(name='Value')
df['Month'] = df.index.month

# Seasonal Subseries Plot
month_plot(df['Value'], ax=axes[2]); axes[2].set_title("Seasonal Subseries Plot");

# %% [markdown]
# - Let's look at the real data now.

# %%
_, ax = plt.subplots(1,1, figsize=(7,3))
month_plot(monthly_temp, ax=ax)
plt.tight_layout();

# %% [markdown]
# - Notice that a `violinplot` can give a very similar information to the `month_plot`.

# %%
_, ax = plt.subplots(1,1, figsize=(8,3))
sns.violinplot(x=monthly_temp.index.month, 
               y=monthly_temp.temp, ax=ax) # notice the indexing on the x by month
plt.grid();

# %% [markdown]
# - Finally, to obtain the numerical value of the main periodicity we can use the Foruier Transfrom.
# - Here, we'll use the function we defined in the first lecture.

# %%
dominant_period, _, _ = fft_analysis(monthly_temp['temp'].values)
print(f"Dominant period: {np.round(dominant_period)}")

# %% [markdown]
# #### Remove the main seasonality
# 
# - In this case, it is clear that the main seasonality is $L=12$.
# - We can remove it with a seasonal differencing.

# %%
monthly_temp['Seasonally_Differenced'] = monthly_temp['temp'].diff(12)

# %%
# Drop nan
monthly_temp_clean = monthly_temp.dropna()
monthly_temp_clean 

# %% [markdown]
# > **⚙ Try it yourself**
# >
# > Try redoing the previous plots on the differenced data!

# %% [markdown]
# #### Identifying $p$ and $q$
# 
# As we learned in the previous chapter, we will identify the AR order $p$ and the MA order $q$ with:
# 
# - Autocorrelation function (ACF) plot.
# - Partial autocorrelation function (PACF) plot.

# %% [markdown]
# **AR($p$)**
# 
# - The order of the AR model is identified as follows:
#     - Plot 95% confidence interval on the PACF (done as default by statsmodels).
#     - Choose lag $p$ such that the partial autocorrelation becomes insignificant for $p+1$ and beyond.

# %% [markdown]
# - If a process depends on previous values of itself then it is an AR process. 
# - If it depends on previous errors than it is an MA process.
# - An AR process propagates shocks infinitely.
# - AR processes will exhibit exponential decay in ACF and a cut-off in PACF.

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 3))
plot_pacf(monthly_temp_clean['Seasonally_Differenced'], lags=20, ax=ax); 

# %% [markdown]
# - It looks like the PACF becomes zero at lag 2.
# - However there is a non-zero partial autocorrelation at lag 3.
# - The optimal value might be $p=1$, $p=2$, or $p=3$.
# - Note that there are high partial autocorrelations at higher lags, especially 12. 
#     - This is an effect from seasonality and seasonal differencing. 
#     - It should not be accounted for when choosing $p$.

# %% [markdown]
# **MA($q$)**
# 
# - The order of the MA model is identified as follows:
#     - Plot 95% confidence interval on the ACF (default in statsmodels).
#     - Choose lag $q$ such that ACF becomes statistically zero for $q+1$ and beyond.

# %% [markdown]
# - MA models do not propagate shocks infinitely; they die after $q$ lags.
# - MA processes will exhibit exponential decay in PACF and a cut-off in ACF.

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 3))
plot_acf(monthly_temp_clean['Seasonally_Differenced'], lags=20, ax=ax); 

# %% [markdown]
# - Also in this case there are non-zero autocorrelations at lags 1 and 3.
# - So, the values to try are $q=1$, $q=2$, or $q=3$.

# %% [markdown]
# ### Model estimation
# 
# - Once the orders $p$ and $q$ are identified is necessary to estimate parameters $\phi_1, \dots, \phi_p$ of the AR part and parameters $\theta_1, \dots, \theta_q$ of the MA part.
# - Estimating the parameters of an ARMA model is a complicated, nonlinear problem.
# - Nonlinear least squares and maximum likelihood estimation are common approaches.
# - Software packages will fit the ARMA model for us.

# %% [markdown]
# - We split the data in two parts:
#     - the training set, that will be used to fit the model's parameters.
#     - the test set, that will be used later on to evaluate the prediction performance of the model on unseen data.

# %%
train = monthly_temp_clean['Seasonally_Differenced'][:-36]
test = monthly_temp_clean['Seasonally_Differenced'][-36:]

plt.figure(figsize=(12,3))
plt.plot(train)
plt.plot(test);

# %%
model = ARIMA(train, order=(3, 0, 3))  # ARIMA with d=0 is equivalent to ARMA
fit_model = model.fit()

print(fit_model.summary())

# %% [markdown]
# ### ARMA Model Validation
# 
# - How do we know if our ARMA model is good enough?
# - We can check the residuals, i.e., what the model was not able to fit.
# - The residuals should approximate a Gaussian distribution (aka white noise).
# - Otherwise, we might need to select a better model.

# %%
residuals = fit_model.resid

plt.figure(figsize=(10,3))
plt.plot(residuals)
plt.title("Residuals");

# %% [markdown]
# > 🤔 How to test if the residuals look like noise?
# 
# - We will use both visual inspection and statistical tests.
# - Visual inspection:
#     - ACF plot.
#     - Histogram.
#     - QQ plot.
# - Statistical tests:
#     - Normality.
#     - Autocorrelation.
#     - Heteroskedasticity.

# %% [markdown]
# #### Visual inspection 
# 
# **ACF plot**
# 
# - Checks for any autocorrelation in the residuals. 
# - White noise should show no significant autocorrelation at all lags.

# %%
_, ax = plt.subplots(1, 1, figsize=(5, 3))
plot_acf(residuals, lags=10, ax=ax)
plt.title('ACF of Residuals');

# %% [markdown]
# **Histogram and QQ-Plot**
# - Assess the normality of the residuals. 
# - White noise should ideally follow a normal distribution.

# %%
plt.figure(figsize=(8,3))
plt.hist(residuals, bins=20, density=True, alpha=0.6, color='g') # Histogram
# Add the normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Results: mu = %.2f,  std = %.2f" % (np.mean(residuals), np.std(residuals))
plt.title(title);

# %%
# QQ-Plot
_, ax = plt.subplots(1, 1, figsize=(6, 6))
qqplot(residuals, line='s', ax=ax)
plt.title('QQ-Plot of Residuals');

# %% [markdown]
# - The plots are conveniently summarized in the function ``plot_diagnostics()`` that can be called on the fit model.

# %%
fit_model.plot_diagnostics(figsize=(10, 6))
plt.tight_layout();

# %% [markdown]
# #### Statistical tests
# 
# **Normality: Jarque-Bera and Shapiro-Wilk tests**
# 
# > $H_0$: the residuals are normally distributed.

# %%
norm_val, norm_p, skew, kurtosis = fit_model.test_normality('jarquebera')[0]
print('Normality (Jarque-Bera) p-value:{:.3f}'.format(norm_p))

# %%
shapiro_test = stats.shapiro(residuals)
print(f'Normality (Shapiro-Wilk) p-value: {shapiro_test.pvalue:.3f}')

# %% [markdown]
# - The small $p$-values allow us to reject $H_0$.
# - Conclusion: the residuals are **not** normally distributed.
# - Is this a contraddiction?

# %% [markdown]
# ```{Note}
# - For reference, let's see what these tests say about data that are actually normally distributed.
# - Try executing the cell below multiple times and see how much the results changes each time.
# - These tests start to be reliable only for large sample sizes ($N>5000$).
# ```

# %%
# generate random normal data
normal_data = np.random.normal(loc=0, scale=1, size=1000)

jb_test = stats.jarque_bera(normal_data)
print(f'Normality (Jarque-Bera) p-value: {jb_test.pvalue:.3f}')

shapiro_test = stats.shapiro(normal_data)
print(f'Normality (Shapiro-Wilk) p-value: {shapiro_test.pvalue:.3f}')

# %% [markdown]
# **Autocorrelation: Ljung-Box test**
# 
# > $H_0$: the residuals are independently distributed (no autocorrelation).
# 
# - There is a $p$-value for each lag. 
# - Here we just take the mean, but one might also want to look at the at largest lag (`pval[-1]`).
# - It is also not always obvious to select how many lags should be used in the test...

# %%
statistic, pval = fit_model.test_serial_correlation(method='ljungbox', lags=10)[0]
print(f'Ljung-Box p-value: {pval.mean():.3f}') 

# %% [markdown]
# **Autocorrelation: Durbin Watson test**
# 
# - Tests autocorrelation in the residuals.
# - We want something between 1-3.
# - 2 is ideal (no serial correlation).

# %%
durbin_watson = ss.stats.stattools.durbin_watson(
    fit_model.filter_results.standardized_forecasts_error[0, fit_model.loglikelihood_burn:])
print('Durbin-Watson: d={:.2f}'.format(durbin_watson))

# %% [markdown]
# **Heteroskedasticity test**
# 
# - Tests for change in variance between residuals.
# > $H_0$: no heteroskedasticity. 
# - $H_0$ indicates different things based on the alternative $H_A$:
#     - $H_A$: Increasing, $H_0$: the variance is not increasing throughout the series.
#     - $H_A$: Decreasing, $H_0$: the variance is not decreasing throughout the series.
#     - $H_A$: Two-sided (default), $H_0$: the variance does not increase nor decrease throughout the series.

# %%
_, pval = fit_model.test_heteroskedasticity('breakvar', alternative='increasing')[0]
print(f'H_a: Increasing - pvalue:{pval:.3f}')

# %%
_, pval = fit_model.test_heteroskedasticity('breakvar', alternative='decreasing')[0]
print(f'H_a: Decreasing - pvalue:{pval:.3f}')

# %%
_, pval = fit_model.test_heteroskedasticity('breakvar', alternative='two-sided')[0]
print(f'H_a: Two-sided - pvalue:{pval:.3f}')

# %% [markdown]
# **Summary of our tests**
# 
# Independence:
# - &#x2705; ACF plot.
# - &#x2705; Ljung-Box test.
# - &#x2705; Durbin Watson test.

# %% [markdown]
# Normality:
# - &#x2705; Histogram/Density plot.
# - 🤔 QQ-plot
# - &#x274C; Jarque-Bera (reliable for large sample size).
# - &#x274C; Shapiro-Wilk (reliable for large sample size).
# 
# Heteroskedasticity
# - &#x274C; Heteroskedasticity test.

# %% [markdown]
# - The tests are a bit inconclusive.
# - There is no strong evidence that the model is either very good or very bad.
# - It is probably wise to try other candidate models, e.g., `ARMA(2,0,2)`, and repeat the tests.

# %% [markdown]
# ### ARMA Model Predictions
# 
# - Once the model is fit, we can use it to predict the test data.
# - The predictions come in the form of a distribution.
# - In other words, ARMA performs a *probabilistic forecasting*.
# - The mean (mode) of this distribution correspond to the most likely value and correspond to our forecast.
# - The rest of the distribution can be used to compute confidence intervals.

# %%
pred_summary = fit_model.get_prediction(test.index[0], test.index[-1]).summary_frame()

plt.figure(figsize=(12, 4))
plt.plot(test.index, test, label='Ground Truth')
plt.plot(test.index, pred_summary['mean'], label='Forecast', linestyle='--')
plt.fill_between(test.index, pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'], 
                 color='orange', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.tight_layout();

# %% [markdown]
# ---

# %% [markdown]
# ## ARIMA Model

# %% [markdown]
# - ARIMA stands for Auto Regressive Integrated Moving Average. 
# - ARIMA models have three components:
#     - AR model.
#     - Integrated component (more on this shortly).
#     - MA model.

# %% [markdown]
# - The ARIMA model is denoted ARIMA($p, d, q$).
#     - $p$ is the order of the AR model.
#     - $d$ is the number of times to difference the data.
#     - $q$ is the order of the MA model.
#     - $p$, $d$, and $q$ are nonnegative integers.

# %% [markdown]
# - As we saw previously, differencing nonstationary time series data one or more times can make it stationary. 
# - That’s the role of the integrated (I) component of ARIMA.
# - $d$ is the number of times to perform a lag 1 difference on the data.
#     - $d=0$: no differencing. 
#     - $d=1$: difference once. 
#     - $d=2$: difference twice. 

# %% [markdown]
# - The ARMA model is suitable for *stationary* time series where the mean and variance do not change over time.
# - The ARIMA model effectively models *non-stationary* time series by differencing the data.
# - In practice, ARIMA makes the time series stationary before applying the ARMA model.
# - Let's see it with an example.

# %%
# Generate synthetic stationary data with an ARMA(1,1) process
n = 250
ar_coeff = np.array([1, -0.7]) # The first value (lag 0) is always 1. AR coeff are negated.
ma_coeff = np.array([1, 0.7])  # The first value (lag 0) and is always 1
arma_data = ss.tsa.arima_process.ArmaProcess(ar_coeff, ma_coeff).generate_sample(
    nsample=n, burnin=1000)

# %%
# Generate a synthetic non-stationary data (needs to be differenced twice to be stationary)
t = np.arange(n)
non_stationary_data = 0.05 * t**2 + arma_data  # Quadratic trend

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(non_stationary_data)
axes[0].set_title('Original Data')
axes[1].plot(diff(non_stationary_data, k_diff=1))
axes[1].set_title('1st Differencing')
axes[2].plot(diff(non_stationary_data, k_diff=2))
axes[2].set_title('2nd Differencing')
plt.tight_layout();

# %%
# Fit models to stationary data
arma_model = ARIMA(arma_data[:-20], order=(1, 0, 1)).fit()
arima_model = ARIMA(arma_data[:-20], order=(1, 1, 1)).fit()

plt.figure(figsize=(12, 4))
plt.plot(arma_data[-50:], 'k', label='Stationary Data', linewidth=2)
plt.plot(arma_model.predict(200,250), label='ARMA Fit')
plt.plot(arima_model.predict(200, 250), label='ARIMA Fit')
plt.legend()
plt.title('Stationary Data')
plt.tight_layout();

print(len(arma_model.predict(10)))

# %%
# Fit models to non-stationary data
arma_model = ARIMA(non_stationary_data[:-20], order=(1, 0, 1)).fit()
arima_model = ARIMA(non_stationary_data[:-20], order=(1, 2, 1)).fit()

plt.figure(figsize=(12, 4))
plt.plot(non_stationary_data[-40:], 'k', label='Non-stationary Data', linewidth=2)
plt.plot(arma_model.predict(210,250), label='ARMA Fit')
plt.plot(arima_model.predict(210,250), label='ARIMA Fit')
plt.legend()
plt.title('Non-stationary Data')
plt.tight_layout();

# %% [markdown]
# ---

# %% [markdown]
# ## SARIMA

# %% [markdown]
# - To apply ARMA and ARIMA, we must remove the seasonal component.
# - After computing the predictions we had to put the seasonal component back.
# - It would be convenient to directly work on data with seasonality.

# %% [markdown]
# - SARIMA is an extension of ARIMA that includes seasonal terms.
# - The model is specified as SARIMA $(p, d, q) \times (P, D, Q, s)$:
#   - Regular ARIMA components $(p, d, q)$.
#   - Seasonal components $(P, D, Q, s)$ where:
#     - $P$: Seasonal autoregressive order.
#     - $D$: Seasonal differencing order.
#     - $Q$: Seasonal moving average order.
#     - $s$: Number of time steps for a single seasonal period.

# %% [markdown]
# **How to select the values $s, P, D, Q$?**
# - $s$: 
#     - Is the main seasonality in the data. 
#     - We already know how to find it.

# %% [markdown]
# - $P$ and $Q$: 
#     - A spike at $s$-th lag (and potentially multiples of $s$) should be present in the PACF/ACF plots. 
#     - For example, if $s = 12$, there could be spikes at $(s*n)^{th}$ lags too. 
#     - Pick out the lags with largest spikes as candidates for $P$ or $Q$.

# %% [markdown]
# - $D$: 
#     - Is the number of seasonal differencing required to make the time series stationary. 
#     - Is often determined by trial and error or by examining the seasonally differenced data.

# %% [markdown]
# ```{tip}
# - Before selecting $P$ and $Q$, ensure that the series is seasonally stationary by applying seasonal differencing if needed ($D$). 
# - Look the ACF plot to identify the seasonal moving average order $Q$. 
#   - Look for significant autocorrelations at seasonal lags (multiples of $s$). 
#   - If the ACF plot shows a sharp cut-off in the seasonal lags, it suggests the order of the seasonal MA component ($Q$).
#   
# - Look at the PACF plot to identify the seasonal autoregressive order $P$. 
#   - Look for significant spikes at multiples of the seasonality $s$. 
#   - A sharp cut-off in the PACF at a seasonal lag suggests the order of the seasonal AR terms ($P$) needed.
# ```

# %%
diff_ts = monthly_temp['temp'].diff(periods=12).dropna()

fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1.plot(diff_ts)
ax2 = plt.subplot2grid((2, 2), (1, 0))
plot_pacf(diff_ts, lags=80, ax=ax2)
ax3 = plt.subplot2grid((2, 2), (1, 1))
plot_acf(diff_ts, lags=80, ax=ax3)
plt.tight_layout();

# %%
# fit SARIMA monthly based on helper plots
sar = ss.tsa.statespace.sarimax.SARIMAX(monthly_temp[:750].temp, 
                                order=(2,1,2), 
                                seasonal_order=(0,1,1,12), 
                                trend='c').fit(disp=False)
sar.summary()

# %%
sar.plot_diagnostics(figsize=(14, 8))
plt.tight_layout();

# %%
monthly_temp['forecast'] = sar.predict(start = 750, end= 792, dynamic=False)  
monthly_temp[730:][['temp', 'forecast']].plot(figsize=(12, 3))
plt.tight_layout();
print(f"MSE: {mse(monthly_temp['temp'][-42:],monthly_temp['forecast'][-42:]):.2f}")

# %% [markdown]
# ## AutoARIMA

# %% [markdown]
# - At this point it should be clear that identifying the optimal SARIMA model is difficult.
# - It requires careful analysis, trial and errors, and some experience.
# - The following cheatsheet summarizes some rules of thumb to select the model.

# %% [markdown]
# **Cheatsheet for coefficients setting**
# 
# <center>
#     
# |ACF Shape|Indicated Model|
# |---|:---|
# |Exponential, decaying to zero|AR model. Use the PACF to identify the order of the AR model.|
# |Alternating positive and negative, decaying to zero|AR model. Use the PACF to identify the order.|
# |One or more spikes, rest are essentially zero|MA model, order identified by where plot becomes zero.|
# |Decay, starting after a few lags|Mixed AR and MA (ARMA) model.|
# |All zero or close to zero|Data are essentially random.|
# |High values at fixed intervals|Include seasonal AR term.|
# |No decay to zero|Series is not stationary.|
#     
# </center>

# %% [markdown]
# - An alternative to manual model selection is to use automated procedures.
# - Here enters AutoARIMA.
# - AutoARIMA requires you to specify the maximum range of values to try.
# - Afterwards, it tries to find the best confgiuration among the possible ones.
# - See [here](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html) for a complete list and description of the options available.

# %%
# Split the data into train and test sets
train, test = monthly_temp[:750].temp, monthly_temp[750:].temp

# Use auto_arima to find the best ARIMA model
model = pm.auto_arima(train, 
                      start_p=0, start_q=0,
                      test='adf',       # Use adftest to find optimal 'd'
                      max_p=2, max_q=2, # Maximum p and q
                      m=12,             # Seasonality
                      start_P=0, start_Q=0,
                      max_P=2, max_Q=2, # Maximum p and q
                      seasonal=True,    # Seasonal ARIMA
                      d=None,           # Let model determine 'd'
                      D=1,              # Seasonal difference
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise search to find the best model

# %%
# Summarize the model
print(model.summary())

# Forecast future values
monthly_temp['forecast'] = model.predict(n_periods=len(test)) 
monthly_temp[730:][['temp', 'forecast']].plot(figsize=(12, 3))
plt.tight_layout();
print(f"MSE: {mse(monthly_temp['temp'][-42:],monthly_temp['forecast'][-42:]):.2f}")

# %% [markdown]
# ### AutoARIMA or not AutoARIMA?
# 
# While being very convenient, like all automated procedures `auto_arima` comes with drawbacks.

# %% [markdown]
# 1. `auto_arima` can be *computationally expensive*, especially for large datasets and when exploring a wide range of models. 

# %% [markdown]
# 2. Automated model selection lacks the qualitative insights a human might bring to the modeling process.
#    - These, include understanding business cycles, external factors, or anomalies in the data. 

# %% [markdown]
# 3. The defaults in `auto_arima` may not be optimal for all time series data. 
#    - The range of values to explore should be adjusted properly each time.
#    - This might become almost as tricky as doing manual model selection.

# %% [markdown]
# 4. `auto_arima` requires a sufficiently long time series to accurately identify patterns and seasonality. 

# %% [markdown]
# 5. The selection of the best model is typically based on *statistical criteria* such as AIC or BIC.
#    - These, might not always align with practical performance metrics such as MSE.

# %% [markdown]
# ## Grid search

# %% [markdown]
# - Selecting the best ARIMA model is as much an art as it is a science, involving iteration and refinement.
# - A common approach is to select a set of candidates for $(p,d,q)\times (P,D,Q,s)$ and fit a model for each possible combination.

# %% [markdown]
# - For each fit model:
#     - Analyze the residuals using the visual techniques and the statistical tests discussed before.
#     - Evaluate the prediction performance. 🤔 How?
#     - Evaluate its complexity. 🤔 How?

# %% [markdown]
# ### Prediction performance 
# 
# Use MSE and/or MAPE to evaluate the prediction performance of the model.

# %% [markdown]
# **Mean Squared Error (MSE)**
# 
# - MSE is the average of the squared differences between the observed values and the predictions.
# - $MSE = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$, where $Y_i$ is the actual observation and $\hat{Y}_i$ is the forecasted value.
# - In SARIMA model selection, a model with a lower MSE is preferred, indicating better fit to the data.

# %% [markdown]
# **Mean Absolute Percentage Error (MAPE)**
# 
# - MAPE is the average of the absolute percentage errors of forecasts.
# - $MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{Y_i - \hat{Y}_i}{Y_i}\right|$, where $Y_i$ is the actual observation and $\hat{Y}_i$ is the forecasted value.
# - MAPE expresses errors as a percentage, making it straightforward to understand the magnitude of forecasting errors.
# - If you are comparing models that predict different quantities (e.g., dollars vs. units sold), the percentage error allows for a more apples-to-apples comparison.
# - Also, when you're more interested in the relative size of the errors than in their absolute size, MAPE is relevant. 
# - Finally, MAPE is useful when the magnitude of the data varies significantly.

# %% [markdown]
# ### Model complexity
# 
# Use AIC or BIC to estimate the model's complexity.

# %% [markdown]
# **Akaike Information Criterion (AIC)**
# 
# - AIC is a measure of the relative quality of statistical models for a given set of data. 
# - It deals with the trade-off between the goodness of fit of the model and the complexity of the model.
# - $AIC = 2k - 2\ln(\hat{L})$, where $k$ is the number of parameters in the model, and $\hat{L}$ is the maximized value of the likelihood function for the model.
# - The model with the lowest AIC value is preferred, as it fits the data well but is not overly complex.

# %% [markdown]
# **Bayesian Information Criterion (BIC)**
# 
# - Similar to AIC, the BIC is another criterion for model selection, but it introduces a stronger penalty for models with more parameters.
# - $BIC = \ln(n)k - 2\ln(\hat{L})$, where $n$ is the number of observations, $k$ is the number of parameters, and $\hat{L}$ is the maximized likelihood.
# - A lower BIC value indicates a better model, preferring simpler models to complex ones, especially as the sample size $n$ increases.

# %% [markdown]
# ### Restricting the search with Exploratory Data Analysis (EDA)
# 
# - Grid search can be very expensive if done exaustively, especially on limited hardware.
# - An Exploratory Data Analysis can help to significantly reduce the number of candidates to try out.

# %% [markdown]
# #### Selecting the candidates for differentiation
# 
# - Let's start by identifying all the candidates for seasonal and general differencing.
# - In this case, we already know that the main seasionality is $s=12$. 
# - Should we apply first the general or the seasonal differencing?

# %% [markdown]
# - If seasonal patterns are dominant and the goal is to remove seasonality before addressing any trend, start with **seasonal differencing**. 
#     - This is particularly useful when the seasonal pattern is strong and clear.
# - If the trend is the predominant feature, you might start with **standard differencing**.

# %% [markdown]
# - In our case, we go with seasonal differencing first.

# %%
# create all combinations of differencing orders, applying seasonal differencing first and then general differencing
def differencing(timeseries, s, D_max=2, d_max=2):
        
    # Seasonal differencing from 0 to D_max
    seas_differenced = []
    for i in range(D_max+1):
        timeseries.name = f"d0_D{i}_s{s}"
        seas_differenced.append(timeseries)
        timeseries = timeseries.diff(periods=s)
    seas_df = pd.DataFrame(seas_differenced).T

    # General differencing from 0 to d_max
    general_differenced = []
    for j, ts in enumerate(seas_differenced):
        for i in range(1,d_max+1):
            ts = ts.diff()
            ts.name = f"d{i}_D{j}_s{s}"
            general_differenced.append(ts)       
    gen_df = pd.DataFrame(general_differenced).T
    
    # concatenate seasonal and general differencing dataframes
    return pd.concat([seas_df, gen_df], axis=1)

# %%
# create the differenced series
diff_series = differencing(monthly_temp['temp'], s=12, D_max=2, d_max=2)
diff_series

# %% [markdown]
# #### Filter-out non-stationary candidates
# 
# - Among all the differenced time series, keep only those that are stationary (according to ADF).

# %%
# create a summary of test results of all the series
def adf_summary(diff_series):
    summary = []
    
    for i in diff_series:
        # unpack the results
        a, b, c, d, e, f = adfuller(diff_series[i].dropna())
        g, h, i = e.values()
        results = [a, b, c, d, g, h, i]
        summary.append(results)
    
    columns = ["Test Statistic", "p-value", "#Lags Used", "No. of Obs. Used",
               "Critical Value (1%)", "Critical Value (5%)", "Critical Value (10%)"]
    index = diff_series.columns
    summary = pd.DataFrame(summary, index=index, columns=columns)
    
    return summary

# %%
# create the summary
summary = adf_summary(diff_series)

# filter away results that are not stationary
summary_passed = summary[summary["p-value"] < 0.05]
summary_passed

# %%
# output indices as a list
index_list = pd.Index.tolist(summary_passed.index)

# use the list as a condition to keep stationary time-series
passed_series = diff_series[index_list].sort_index(axis=1)

# %%
# Plot the final set of time series 
# NOTE: these plots are too small. Make a larger plot for each series to see things better!
fig, axes = plt.subplots(3, 3, figsize=(30, 10), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    passed_series.iloc[:,i].plot(ax=ax)
    ax.set_title(passed_series.columns[i])
    ax.grid()

# %% [markdown]
# #### Select candidates for $p, q, P, Q$
# 
# - We are going to make a script to extract candidates for the orders of the AR and MA component, both in the general and the seasonal part.
# - We will leverage the ACF and PACF functions.
# - So far, we looked at the `acf_plot` and `pacf_plot`.
# - Now we need to use `acf` and `pacf`.
# - We need to understand how these relate.

# %%
PACF, PACF_ci = pacf(passed_series.iloc[:,0].dropna(), alpha=0.05)

# Plot PACF
plt.figure(figsize=(10,3))
plt.plot(PACF, color='k', label='PACF')
plt.plot(PACF_ci, color='tab:blue', linestyle='--', label=['95% Confidence Interval', ''])
plt.legend()
plt.tight_layout();

# %%
# subtract the confidence interval from the PACF to center the CI in zero
plt.figure(figsize=(10,3))
plt.fill_between(range(29), PACF_ci[:,0] - PACF, PACF_ci[:,1] - PACF, color='tab:blue', alpha=0.3)
plt.hlines(y=0.0, xmin=0, xmax=29, linewidth=1, color='gray')

# Display the PACF as bars
plt.vlines(range(29), [0], PACF[:29], color='black')
plt.tight_layout();

# %%
df_sp_p = pd.DataFrame() # create an empty dataframe to store values of significant spikes in PACF plots
for i in passed_series:
    # unpack the results into PACF and their CI
    PACF, PACF_ci = pacf(passed_series[i].dropna(), alpha=0.05, method='ywm')
    # subtract the upper and lower limits of CI by PACF to centre CI at zero
    PACF_ci_ll = PACF_ci[:,0] - PACF
    PACF_ci_ul = PACF_ci[:,1] - PACF
    # find positions of significant spikes representing possible value of p & P
    sp1 = np.where(PACF < PACF_ci_ll)[0]
    sp2 = np.where(PACF > PACF_ci_ul)[0]
    # PACF values of the significant spikes
    sp1_value = abs(PACF[PACF < PACF_ci_ll])
    sp2_value = PACF[PACF > PACF_ci_ul]
    # store values to dataframe
    sp1_series = pd.Series(sp1_value, index=sp1)
    sp2_series = pd.Series(sp2_value, index=sp2)
    df_sp_p = pd.concat((df_sp_p, sp1_series, sp2_series), axis=1)
df_sp_p = df_sp_p.sort_index() # Sort the dataframe by index
# visualize sums of values of significant spikes in PACF plots ordered by lag
df_sp_p.iloc[1:].T.sum().plot(kind='bar', title='Candidate AR Terms', xlabel='nth lag', ylabel='Sum of PACF', figsize=(8,3));

# %%
df_sp_q = pd.DataFrame()
for i in passed_series:
    # unpack the results into ACF and their CI
    ACF, ACF_ci = acf(passed_series[i].dropna(), alpha=0.05)
    # subtract the upper and lower limits of CI by ACF to centre CI at zero
    ACF_ci_ll = ACF_ci[:,0] - ACF
    ACF_ci_ul = ACF_ci[:,1] - ACF
    # find positions of significant spikes representing possible value of q & Q
    sp1 = np.where(ACF < ACF_ci_ll)[0]
    sp2 = np.where(ACF > ACF_ci_ul)[0]
    # ACF values of the significant spikes
    sp1_value = abs(ACF[ACF < ACF_ci_ll])
    sp2_value = ACF[ACF > ACF_ci_ul]
    # store values to dataframe
    sp1_series = pd.Series(sp1_value, index=sp1)
    sp2_series = pd.Series(sp2_value, index=sp2)
    df_sp_q = pd.concat((df_sp_q, sp1_series, sp2_series), axis=1)
df_sp_q = df_sp_q.sort_index() # Sort the dataframe by index
# visualize sums of values of significant spikes in ACF plots ordered by lags
df_sp_q.iloc[1:].T.sum().plot(kind='bar', title='Candidate MA Terms', xlabel='nth lag', ylabel='Sum of ACF', figsize=(8,3));

# %% [markdown]
# #### Define the grid of values to search

# %%
# possible values
p = [1, 2, 3]
d = [0, 1]
q = [1, 2]
P = [0, 1]
D = [0, 1, 2]
Q = [0, 1]
s = [12]

# create all combinations of possible values
pdq = list(product(p, d, q))
PDQm = list(product(P, D, Q, s))

print(f"Number of total combinations: {len(pdq)*len(PDQm)}")

# %% [markdown]
# #### Train the models
# 
# - We defined a function that takes every model configuration and trains a model.
# - For each model, we save the MSE, MAPE, AIC and BIC.

# %%
warnings.simplefilter("ignore")
def SARIMA_grid(endog, order, seasonal_order):

    # create an empty list to store values
    model_info = []
       
    #fit the model
    for i in tqdm(order):
        for j in seasonal_order:
            try:
                model_fit = SARIMAX(endog=endog, order=i, seasonal_order=j).fit(disp=False)
                predict = model_fit.predict()
            
                # calculate evaluation metrics: MAPE, RMSE, AIC & BIC
                MAPE = (abs((endog-predict)[1:])/(endog[1:])).mean()
                MSE = mse(endog[1:], predict[1:])
                AIC = model_fit.aic
                BIC = model_fit.bic
            
                # save order, seasonal order & evaluation metrics
                model_info.append([i, j, MAPE, MSE, AIC, BIC])
            except:
                continue
                
    # create a dataframe to store info of all models
    columns = ["order", "seasonal_order", "MAPE", "MSE", "AIC", "BIC"]
    model_info = pd.DataFrame(data=model_info, columns=columns)
    return model_info

# %%
# create train-test-split
train = monthly_temp['temp'].iloc[:int(len(monthly_temp)*0.9)]
test = monthly_temp['temp'].iloc[int(len(monthly_temp)*0.9):]

# %%
start = time.time()

# fit all combinations into the model
model_info = SARIMA_grid(endog=train, order=pdq, seasonal_order=PDQm)

end = time.time()
print(f'time required: {end - start :.2f}')

# %% [markdown]
# #### Analyze the results
# 
# - Show the 10 best models according to the performance (MSE, MAPE) and model complexity (AIC, BIC).

# %%
# 10 least MAPE models
least_MAPE = model_info.nsmallest(10, "MAPE")
least_MAPE

# %%
# 10 least MSE models
least_MSE = model_info.nsmallest(10, "MSE")
least_MSE

# %%
# 10 least AIC models
least_AIC = model_info.nsmallest(10, "AIC")
least_AIC

# %%
# 10 least BIC models
least_BIC = model_info.nsmallest(10, "BIC")
least_BIC 

# %% [markdown]
# - We can check if there are overlaps between the 4 different groups using the `set` function.

# %%
set(least_MAPE.index) & set(least_MSE.index)

# %%
set(least_AIC.index) & set(least_BIC.index)

# %%
set(least_MSE.index) & set(least_AIC.index)

# %% [markdown]
# - Show the top model according to each metric.

# %%
# the best model by each metric
L1 = model_info[model_info.MAPE == model_info.MAPE.min()]
L2 = model_info[model_info.MSE == model_info.MSE.min()]
L3 = model_info[model_info.AIC == model_info.AIC.min()]
L4 = model_info[model_info.BIC == model_info.BIC.min()]

best_models = pd.concat((L1, L2, L3, L4))
best_models

# %% [markdown]
# #### Compute performance on the test set
# 
# - Take the best models, compute the predictions and evaluate their performance in terms of MAPE on the w.r.t. test data.

# %%
# Take the configurations of the best models
ord_list = [tuple(best_models.iloc[i,0]) for i in range(best_models.shape[0])]
s_ord_list = [tuple(best_models.iloc[i,1]) for i in range(best_models.shape[0])]
preds, ci_low, ci_up, MAPE_test = [], [], [], []

# Fit the models and compute the forecasts
for i in range(4):
    model_fit = SARIMAX(endog=train, order=ord_list[i], 
                        seasonal_order=s_ord_list[i]).fit(disp=False) # Fit the model
    pred_summary = model_fit.get_prediction(test.index[0], 
                                            test.index[-1]).summary_frame() # Compute preds
    # Store results
    preds.append(pred_summary['mean'])
    ci_low.append(pred_summary['mean_ci_lower'][test.index])
    ci_up.append(pred_summary['mean_ci_upper'][test.index])
    MAPE_test.append((abs((test-pred_summary['mean'])/(test)).mean()))

# %%
# visualize the results of the fitted models
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24,6), sharex=True, sharey=True)
titles = [f'Least MAPE Model {ord_list[0]} x {s_ord_list[0]}',
          f'Least MSE Model {ord_list[1]} x {s_ord_list[1]}',
          f'Least AIC Model {ord_list[2]} x {s_ord_list[2]}',
          f'Least BIC Model {ord_list[3]} x {s_ord_list[3]}']
k = 0
for i in range(2):
    for j in range(2):
        axs[i,j].plot(monthly_temp['temp'], label='Ground Truth')
        axs[i,j].plot(preds[k], label='Prediction')
        axs[i,j].set_title(titles[k] + f' -- MAPE test: {MAPE_test[k]:.2%}')
        axs[i,j].legend()
        axs[i,j].axvline(test.index[0], color='black', alpha=0.5, linestyle='--')
        axs[i,j].fill_between(x=test.index, y1=ci_low[k], y2=ci_up[k], color='orange', alpha=0.2)
        axs[i,j].set_ylim(bottom=20, top=90)
        axs[i,j].set_xlim(left=monthly_temp.index[-120], right=monthly_temp.index[-1])
        k += 1
plt.tight_layout()
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ## Summary
# 
# 1. Autoregressive Moving Average (ARMA) models.
# 2. Autoregressive Integrated Moving Average (ARIMA) models.
# 3. SARIMA models (ARIMA model for data with seasonality).
# 4. Selecting the best model.
# 
# ---

# %% [markdown]
# ## Exercise
# 
# - Look at sensor data that tracks atmospheric CO2 from continuous air samples at Mauna Loa Observatory in Hawaii. This data includes CO2 samples from MAR 1958 to DEC 1980.

# %%
co2 = pd.read_csv('https://zenodo.org/records/10951538/files/arima_co2.csv?download=1', 
                  header = 0,
                  names = ['idx', 'co2'],
                  skipfooter = 2)

# convert the column idx into a datetime object and set it as the index
co2['idx'] = pd.to_datetime(co2['idx'])
co2.set_index('idx', inplace=True)

# Rmove the name "idx" from the index column
co2.index.name = None
co2

# %% [markdown]
# - Determine the presence of main trend and seasonality in the data.
# - Determine if the data are stationary.
# - Split the data in train (90%) and test (10%).
# - Find a set of SARIMAX candidate models by looking at the ACF and PACF.
# - Perform a grid search on the model candidates.
# - Select the best models, based on performance metrics, model complexity, and normality of the residuals.
# - Compare the best model you found with the one from autoarima.


