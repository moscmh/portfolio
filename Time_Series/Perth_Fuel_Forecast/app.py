import streamlit as st

st.set_page_config(page_title=None, 
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="expanded", 
                   menu_items=None)

st.title("Perth Fuel Price Forecast")

# Sidebar
st.sidebar.header("Content")
button_intro = st.sidebar.button("Introduction")
button_eda = st.sidebar.button("Exploratory Data Analysis")
button_ARIMA = st.sidebar.button("ARIMA Model")
button_SARIMA = st.sidebar.button("SARIMA Model")
button_SARIMAX = st.sidebar.button("SARIMAX Model")

buttons = [button_intro, button_eda, button_ARIMA, button_SARIMA, button_SARIMAX]

# Default Page
if True not in buttons:
    st.header("Introduction")
    st.image("header.jpg", caption="photo from The West Australian")
    st.markdown("&emsp;The project follows `Box-Jenkins`'s modelling framework that consists of `identification`, `estimation`, `model diagnostic`, and `production` to practise analysing and making forecast of a time series dataset, Perth's fuel prices. Note that the fuel price is expressed as cents per litre (AUD/100L). Data was collected from [Fuel Watch](https://www.fuelwatch.wa.gov.au).") 
    st.markdown("&emsp;According to the forecast, the average fuel price in Perth metro area will fluctuate around `AUD$177/100L` in the coming two years. The forecast does not reproduce the spikes as happened in the past. To improve, more data from the past is recommended to determine if there is a pattern and/or a trend in the time series.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Import")
        st.markdown("MySQL")
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Time Series Analysis")
        st.markdown("Adfuller, ARIMA, SARIMAX")

# Introduction
if button_intro:
    st.header("Introduction")
    st.image("header.jpg", caption="photo from The West Australian")
    st.markdown("&emsp;The project follows `Box-Jenkins`'s modelling framework that consists of `identification`, `estimation`, `model diagnostic`, and `production` to practise analysing and making forecast of a time series dataset, Perth's fuel prices. Note that the fuel price is expressed as cents per litre (AUD/100L). Data was collected from [Fuel Watch](https://www.fuelwatch.wa.gov.au).") 
    st.markdown("&emsp;According to the forecast, the average fuel price in Perth metro area will fluctuate around `AUD$177/100L` in the coming two years. The forecast does not reproduce the spikes as happened in the past. To improve, more data from the past is recommended to determine if there is a pattern and/or a trend in the time series.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Import")
        st.markdown("MySQL")
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Time Series Analysis")
        st.markdown("Adfuller, ARIMA, SARIMAX")

# Exploratory Data Analysis
if button_eda:
    st.header("Exploratory Data Analysis")
    st.write("The important parts of the exploratory data analysis are listed in the following.")

    with st.expander("Stationarity"):
        st.markdown("![Original Fuel Price](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/original_price.png?raw=true)")
        st.markdown("* Original fuel price is stationary as shown in the plot as well as Adfuller's test where `p` value is `.0006`.")
        st.markdown("* Differencing and log-return of fuel price are therefore unnecessary.")

    with st.expander("Orders of ARIMA"):
        st.markdown("![ACF and PACF for ARIMA](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_ACF_PACF.png?raw=true)")
        st.markdown("* Both ACF and PACF plots also show inconclusive orders for ARIMA model as there is no sign of tailing off.")
        st.markdown("* By searching through different combinations of the p and q values, the model with (0,d,2) yields the best AIC value.")

# ARIMA Model
if button_ARIMA:
    st.header("ARIMA Model")
    st.markdown("![ARIMA model diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_diagnostics.png?raw=true)")
    st.markdown("* The mean absolute error is 7.145.")
    st.markdown("* The diagnostics, however, indicate that the model has much room to improve. There is a pattern shown in the `Normal Q-Q` plot and the distribution of the residuals does not resemble closely a normal distribution.")    
    st.markdown("* Nevertheless, the Prob(Q) and Prob(JB) are both non-significant. They contradicts the plots.")
    st.markdown("![ARIMA One-Step Ahead In-Sample Prediction](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/arima_1_insample.png?raw=true)")
    st.markdown("* For `One-Step Ahead In-Sample Prediction`, the model is able to capture the movement but the values are much closer to the average.")
    st.markdown("![ARIMA Model Validation](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/ARIMA_valid.png?raw=true)")
    st.markdown("* However, the model does not have enough evidence to make accurate prediction.")

# SARIMA Model
if button_SARIMA:
    st.header("SARIMA Model")
    st.markdown("* Period of 3 is used for modelling with seasonal pattern.")
    st.markdown("![SARIMA Diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMA_diagnostics.png?raw=true)")
    st.markdown("* `pmdarima` searches and finds the best model with (2,0,0)(0,1,0)[3] as both the non-seasonal and seasonal orders.")
    st.markdown("* The data points on `Normal Q-Q` lies closer to the centre line and the residuals distribution is more closer to the normal distribution.")
    st.markdown("* With only the given seasonal factor, the model is unable to capture the trend of the fuel price.")

# SARIMAX Model
if button_SARIMAX:
    st.header("SARIMAX Model")
    st.markdown("&emsp;An additional feature which is maximum temperature is used to try to improve the performance of the model. The data was collected from [Bureau of Meterology](http://www.bom.gov.au/?ref=logo).")
    st.markdown("![SARIMAX Diagnostics](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_diag.png?raw=true)")
    st.markdown("* The diagnostics show that the residuals distribution moderately resembles the normal distribution. There is still a pattern among the data points in the Normal Q-Q plot.")
    st.markdown("![SARIMAX Model Validation](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_valid.png?raw=true)")
    st.markdown("* In the validation, the model cannot follow closely the local downward trend.")
    st.markdown("![SARIMAX Forecast](https://github.com/moscmh/portfolio/blob/main/Time_Series/Perth_Fuel_Forecast/plot/SARIMAX_forecast.png?raw=true)")
    st.markdown("* The future fuel price forecasted by the SARIMAX model fluctuates around AUD$177/100L in the coming two years.")