# Perth Fuel Price Forecast
&emsp;The project follows `Box-Jenkins`'s modelling framework that consists of `identification`, `estimation`, `model diagnostic`, and `production` to practise analysing and making forecast of a time series dataset, Perth's fuel prices. Note that the fuel price is expressed as cents per litre (AUD/100L).  

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
