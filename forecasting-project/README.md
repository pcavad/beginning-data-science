# Forecasting Project for order transactions #
(only for educational purposes, no real data. Please refer to this readme file to see plots and tables).

## Business understanding - main objective of the analysis ##

The business context is to support a customer with forecasting. Their business is seasonal with peak periods for Christmas time and events such as Black Friday. However, being a worldwide distribution different peak seasons may apply (e.g. Chinese New Year, etc). Being the brand less than 5 years the sale channels didn't start at the same time which adds a significant randomness to the data.

## Business goal ##

I ultimately want to predict business and provide a better insight into which factors can affect sales ahead of the time. In a contemporary scenario the sale channllenge is shifting toward a strong interaction with the sourcing channel where brands experience scarsity of components and increased shipping costs. 

## Brief description of the data ##

The dataset includes orders data from one of my customers. Data have been originally dumped from the ERP. The orders file describes transactions between the brand and wholesalers around the world and it was de-normalized to include headers and lines data. It is composed of 9781 rows and it encompasses transactions along 54 months, from July 2016 until June 2021. The fetaures include billing and shipping information as well as line items. I re-organized the data to obtain a time series with totals and sub-stotals for 4 different sale channels.Having in mind to provide reporting and forecasting I developed different assets:

- ETL .py library: routines to fit the raw data for analysis
- Helper .py library: functions which generate pivot tables and plots
- Deep Learning .py library: functions to make a time series and train a simple RNN and a LSTM model
- ETL notebook: calls ETL functions with parameters
- EDA notebook: the place where using the helper library I plot reports and analyse the time series
- Forecasting notebook: the place where I run different time series forecasts

## ETL: ##

- read and load many raw files in csv format
- filter out (cancelled, etc.)
- drop useless columns
- create new columns for analysis
- creates an order date column datetime64 for resampling
- create an exchange rates file to use only USD
- harmonize data as needed (e.g. wholesaler names)
- save the processed dataframe to a new csv file

## EDA: ##

The time series have been originally sampled on a monthly basis to avoid re-sampling and interpolation.

The sale channels didn't start at the same time which became a challenge to obtain a normal distribution:

![channels cumulated plot](img/channels.jpg)

The rolling mean have been also very different along time:

![channels rolling mean](img/channels_rolling_mean.jpg)

The chunks of the mean and variance values of the total time series show a significant variation:

![chunks](img/chunks.jpg)

The rolling means of the total confirm a trend and the standard deviation changes along time, while the boxplot shows few outliers:

![boxplot](img/boxplot.jpg)

The histogram and normal test show a non-normal distribution, while the Dickey-Fuller test is stationary:

![tests timeseries](img/tests_timeseries.jpg)

**Reduce trend and variance**

I run a Log transformation to reduce variance: the result wasn't normally distributed nor stationary:

![log transformation](img/log_transformation.jpg)

I applied Triple Exponential Smoothing: the result was normal but not stationary:

![Triple Exponential Smoothing](img/triple_smoothing.jpg)

**Reduce seasonality**

The decomposition of the time series show a trend, seasonality in both 6 and 12 months, and a residual with a spike around July 2019. The residual is not normal but it is stationary:

![decomposition](img/decomposition1.jpg)

![decomposition](img/decomposition2.jpg)

I went to a differentiated time series and found a normal distribution (below the plot with 2 months diff):

![difference](img/difference.jpg)


**I didn't apply any transformation during the next section in which I run different forecasting models. The original time seris works for the sake of the exercise  without adding complexity.**

## Key findings ##

I have implemented 4 different approaches which can deal with both trend and seasonality:

- ARMA
- SARIMA (with different optimizations)
- Deep Learning with RNN
- Deep Learning with LSTM and LSTM stacked

I used the **Mean Squared Error** as a metric to compare the performance.

**1st step: ARMA**

The acf and pacf plots suggest an autocorrelation model of second order.

![acf plots](img/acf.jpg)

Optimization of the pq parameters: [(2,0), (2,1),(2,2)]

![ARMA](img/arma2.jpg)

**3nd step: SARIMA**

I run the model:

![SARIMA](img/sarima1.jpg)

The diagnostic plots show a barely normal behavior:

![SARIMA](img/sarima2.jpg)

I run additional SARIMA tests:

- Normality: the Null hypothesis is normally distributed residuals
- Ljung-Box: the Null hypothesis is no serial correlation in residuals
- Heteroskedasticity: the null hypothesis is no heteroskedasticity (tests for change in variance between residuals)
- Durbin-Watson: we want between 1-3, 2 is ideal (tests autocorrelation of residuals)

The results for the residuals have been not ambitious:

Test | Result | Interpretation |
--- | --- | --- |
Normality | val=18.104, p=0.000 | no normal distribution |
Ljung-Box | val=1.951, p=0.924 | serial correlation |
Heteroskedasticity | val=0.296, p=0.016 | heteroskedasticity |
Durbin-Watson | d=1.98 | no autocorrelation

I run the auto_feat optimization and came up with parameters used for a second round of modeling:

![ARIMA auto fit](img/arima_auto_fit.jpg)

The diagnostic plots show a normal behavior but the MSE didn't improve in the end:

![ARIMA auto fit](img/tests_arima_auto_fit.jpg)

**4th step: FB Prophet**

The model was able to pick up trend and seasonality without much optimization:

![FB Prophet](img/fb_prophet.jpg)

**5th step: Deep Learning**

I used the support functions to prepare the time series for the Keras processing, and to generate the models. I trained a simple RNN with dense layer and two variations of LSTM.

**RNN**

Fitting and predicting with 70 cells and 1500 epochs. I experimented with different activation functions and different optimizers and I didn't find any striking difference. However, the model picked up the trend fairly well.

![RNN summary](img/rnn_summary.jpg)

![RNN plot](img/rnn.jpg)

**LSTM**

Fitting and predicting with 70 cells and 1500 epochs: 

![LSTM summary](img/lstm_summary.jpg)

![LSTM plot](img/lstm.jpg)

**LSTM stacked**

I modeled and trained a second version with 2 LSTM layers stacked, which picked up the trend slightly better:

![LSTM stacked summary](img/lstm2_summary.jpg)

![LSTM stacked plot](img/lstm2.jpg)

### Plotting actual series vs predictions ### 

![Actual SARIMA](img/actual_sarima.jpg)

![Actual Prophet](img/actual_prophet.jpg)

![Actual RNN](img/actual_rnn.jpg)

![Actual LSTM stacked](img/actual_lstm2.jpg)

![Train Test](img/train_test.jpg)

### Plotting out-of-sample forecasts ###

![Out of sample](img/out_of_sample.jpg)

## Summary ##

FB Prophet ranks well and it offers options to gain insight into the data, more than the RNN which also picked up the trend very well without much of optimization. I found ARIMA and SARIMA more difficult to fine tune, while the LSTMs have been the most challenging. I didn't reach a high degree of optimization which is in the radar for the future. All the neural networks need more work to prepare the data and obtain results. For instance both the RNN and the LSTMs wouldn't pick up any trend without scaling the data.

Model | MSE
------ | ------
ARMA | 6.83423e+08
SARIMA | 2.77195e+08
FB Prophet | 1.06666e+08
RNN | 9.75036e+07
LSTM | 1.35419e+09
LSTM stacked | 1.0253e+09

## Possible flaws in the model and possible improvements ##

My analysis can improve dramatically with more ground work to prepare the time series, and the bottom line is to obtain a stationary series and with the forecast in hands reverse the transformations to obtain real numbers. It is also possible that more processing power can make the neural networks dramatically more efficient, although they remain difficult to interprtate.
