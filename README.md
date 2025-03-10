# Projects Description
## [Medical Cost Prediction](https://ega9jfdpus338or2uiuqx8.streamlit.app/)
&emsp;The [dataset](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) contains `1338` individuals' information with `6` attributes. They are `age`, `sex`, `BMI`, `number of children`, `whether is a smoker`, `residential region`. The outcome variable is the medical cost. This project showed a promising predictive **linear regression** model for estimating medical cost. It can be useful for insurance companies and government to determine policy cost and to allocate resources respectively. 

&emsp;Eventually, after conducting **exploratory data analysis** and **feature engineering**, a new feature was created that may be related to `hospital choice`. Together with `age` and `whether is a smoker`, a regression model with approximately 96.6% accuracy was built.  
* Data cleaning
* Data visualisation
* Exploratory Data Analysis
* Feature engineering
* Linear Regression

## [Perth Fuel Forecast](https://moscmh-portfolio-time-seriesperth-fuel-forecastapp-0rwtuw.streamlit.app)
&emsp;The project follows `Box-Jenkins`'s modelling framework that consists of `identification`, `estimation`, `model diagnostic`, and `production` to practise analysing and making forecast of a time series dataset, Perth's fuel prices. Note that the fuel price is expressed as cents per litre (AUD/100L). Data was collected from [Fuel Watch](https://www.fuelwatch.wa.gov.au).  
* MySQL Database Management
* Data visualisation
* Stationarity Test
* Non-seasonal and seasonal orders search
* Time series `ARIMA` and `SARIMAX` modelling and forecast

## [Pneumonia Detection Model](https://moscmh-portfolio-deep-learningpneumoniaapp-n1ooah.streamlit.app/)
&emsp;Two `CNN` models and a `Transfer Learning` model were built preliminarily. The simplest `CNN` model outperformed the `ResNet101V2` pre-trained model. The accuracies on `test set` were around `85%` and `67%` respectively.

&emsp;Further tuning and regularisation techniques need to be considered in order to improve the models. A model with `99%` accuracy is expected because pneumonia is a serious medical condition with a `concerning mortality rate`.
* Deep Neural Network
* Convolutional Neural Network
* Weight Initialisation
* Regularisation
* Transfer Learning Model
