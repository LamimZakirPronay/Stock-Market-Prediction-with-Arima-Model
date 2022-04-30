import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import statsmodels
from statsmodels.tsa.arima.model import   ARIMA
from statsmodels.tsa.arima.model import  ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log
import pmdarima
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
df =pd.read_csv(r'C:\Users\prona\PycharmProjects\Lasttry\TSLA.csv' ,names = ['Close'], header = 0)
res = adfuller(df.Close.dropna())
print('Augmented Dickey-Fuller Statistic: %f' % res[0])
print('p-value: %f' % res[1])
X = df.Close
# Augmented Dickey Fuller Test
adftest = ndiffs(X, test='adf')
# KPSS Test
kpsstest = ndiffs(X, test='kpss')
# PP Test
pptest = ndiffs(X, test='pp')
print("ADF Test =", adftest)
print("KPSS Test =", kpsstest)
print("PP Test =", pptest)



plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
# The Genuine Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Close);
axes[0, 0].set_title('The Genuine Series')
plot_acf(df.Close, ax=axes[0, 1])


# Order of Differencing: First
axes[1, 0].plot(df.Close.diff());
axes[1, 0].set_title('Order of Differencing: First')
plot_acf(df.Close.diff().dropna(), ax=axes[1, 1])


# Order of Differencing: Second
axes[2, 0].plot(df.Close.diff().diff());
axes[2, 0].set_title('Order of Differencing: Second')
plot_acf(df.Close.diff().diff().dropna(), ax=axes[2, 1])
plt.show()
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Close.diff());
axes[0].set_title('Order of Differencing: First')
axes[1].set(ylim=(0, 5))
plot_pacf(df.Close.diff().dropna(), ax=axes[1])
plt.show()
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
# importing data
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Close.diff());
axes[0].set_title('Order of Differencing: First')
axes[1].set(ylim=(0, 1.2))
plot_acf(df.Close.diff().dropna(), ax=axes[1])
plt.show()
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
# importing data
# Creating ARIMA model
mymodel = ARIMA(df.Close, order=(1, 1, 1))
modelfit = mymodel.fit()#   disp=0
# Plotting Residual Errors
myresiduals = pd.DataFrame(modelfit.resid)
fig, ax = plt.subplots(1, 2)
myresiduals.plot(title="Residuals", ax=ax[0])
myresiduals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
mymodel = ARIMA(df.Close, order=(1, 1, 1))
modelfit = mymodel.fit()#   disp=0
modelfit.get_prediction(start=1,dynamic = False).conf_int().plot()
plt.show()
