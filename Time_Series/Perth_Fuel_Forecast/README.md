# Perth Fuel Price Forecast
&emsp;The project follows `Box-Jenkins`'s modelling framework that consists of `identification`, `estimation`, `model diagnostic`, and `production` to practise analysing and making forecast of a time series dataset, Perth's fuel prices. Note that the fuel price is expressed as cents per litre (AUD/100L). Data was collected from [Fuel Watch](https://www.fuelwatch.wa.gov.au).  

&emsp;According to the forecast, the average fuel price in Perth metro area will fluctuate around `AUD$177/100L` in the coming two years. The forecast does not reproduce the spikes as happened in the past. To improve, more data from the past is recommended to determine if there is a pattern and/or a trend in the time series.

## Python Libraries
### Import Data
MySQL
### Data Manipulations
numpy, pandas
### Visualization Tools
matplotlib, seaborn
### Time Series Analysis
Adfuller, ARIMA, SARIMAX

## Stationarity
![Original Fuel Price](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/original_price.png?raw=true)
* Original fuel price is stationary as shown in the plot as well as Adfuller's test where `p` value is `.0006`.
* Differencing and log-return of fuel price are therefore unnecessary.

## Orders of ARIMA
![ACF and PACF for ARIMA](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_ACF_PACF.png?raw=true)
* Both ACF and PACF plots also show inconclusive orders for ARIMA model as there is no sign of tailing off.
* By searching through different combinations of the p and q values, the model with (0,d,2) yields the best AIC value.

## Model Validation
![ARIMA model diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_diagnostics.png?raw=true)
* The mean absolute error is 7.145.
* The diagnostics, however, indicate that the model has much room to improve. There is a pattern shown in the `Normal Q-Q` plot and the distribution of the residuals does not resemble closely a normal distribution.
* Nevertheless, the Prob(Q) and Prob(JB) are both non-significant. They contradicts the plots.
![ARIMA One-Step Ahead In-Sample Prediction](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/arima_1_insample.png?raw=true)
* For `One-Step Ahead In-Sample Prediction`, the model is able to capture the movement but the values are much closer to the average.
![ARIMA Model Validation](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_valid.png?raw=true)
* However, the model does not have enough evidence to make accurate prediction.

## Seasonal Pattern
* Period of 3 is used for modelling with seasonal pattern.
![SARIMA Diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMA_diagnostics.png?raw=true)
* `pmdarima` searches and finds the best model with (2,0,0)(0,1,0)[3] as both the non-seasonal and seasonal orders.
* The data points on `Normal Q-Q` lies closer to the centre line and the residuals distribution is more closer to the normal distribution.
* With only the given seasonal factor, the model is unable to capture the trend of the fuel price.

## Exogenous Feature
&emsp;An additional feature which is maximum temperature is used to try to improve the performance of the model. The data was collected from [Bureau of Meterology](http://www.bom.gov.au/?ref=logo).
![SARIMAX Diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_diag.png?raw=true)
* The diagnostics show that the residuals distribution moderately resembles the normal distribution. There is still a pattern among the data points in the Normal Q-Q plot.
![SARIMAX Model Validation](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_valid.png?raw=true)
* In the validation, the model cannot follow closely the local downward trend.
![SARIMAX Forecast](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_forecast.png?raw=true)
* The future fuel price forecasted by the SARIMAX model fluctuates around AUD$177/100L in the coming two years.