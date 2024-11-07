# Medical Cost Prediction
&emsp;The [dataset](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) contains `1338` individuals' information with `6` attributes. They are `age`, `sex`, `BMI`, `number of children`, `whether is a smoker`, `residential region`. The outcome variable is the medical cost. This project showed a promising predictive **linear regression** model for estimating medical cost. It can be useful for insurance companies and government to determine policy cost and to allocate resources respectively. 

&emsp;Eventually, after conducting **exploratory data analysis** and **feature engineering**, a new feature was created that may be related to `hospital choice`. Together with `age` and `whether is a smoker`, a regression model with approximately 96.6% accuracy was built.

&emsp;Future work requires researching about the exact feature that was created. 

## Python Libraries
### Data Manipulations
numpy, pandas
### Visualization Tools
matplotlib, seaborn
### Clustering
DBSCAN, GaussianMixture from Scikit-Learn
### Model Building & Evaluation
LinearRegression from Scikit-Learn

# Exploratory Data Analysis
The important parts of the exploratory data analysis are listed in the following.

## Categorical Features
![Categorical Features](https://github.com/moscmh/medicalcost/blob/main/plot/categorical.png?raw=true)
* `Sex` and `residential region` were found to have no significant relationship with `medical cost`.
* `Whether a smoker` showed a significant difference in medical cost between smokers and non-smokers.

## Numeric Features
![Numeric Features](https://github.com/moscmh/medicalcost/blob/main/plot/numeric.png?raw=true)
* Three scatterplots were shown to examine the relationships of `Age` and `BMI` with `medical cost`.
* `BMI` did not show a linear relationship with `medical cost`.
* There were three trendlines in the scatterplot for `medical cost` vs `age`. Clustering was conducted next to efficiently label the three different groups.

## Feature Engineering
![Feature engineering](https://github.com/moscmh/medicalcost/blob/main/plot/feature_engineering.png?raw=true)
* **Guassian Mixture Model** was used to efficiently cluster the individuals from the three trendlines respectively.
* The trendlines could be explained by the choice of hospitals.

# Model Training and Result
![Metrics](https://github.com/moscmh/medicalcost/blob/main/plot/metrics.png?raw=true)
* Eventually, `age`, `BMI`, and the `newly created feature` were used to train a linear regression model.
* The accuracy of the model on training and testing data are `96.7%` and `96.6%` respectively.
* The model can be expressed by the following equation. `Label_1` and `Label_2` are binary values that belong to the categorical `newly created feature`. The sum of the binary values for this 2 labels does not exceed 1. If both are `0`, the individual has `Label_3` as the value of the `newly created feature`.

$$Charges = -3717.00 + 1387.91(Smoker) + 273.81(Age) + 32640.09(Label_1) + 13200.32(Label_2)$$  

* For instance, if an individual is a `smoker`, `50` years old, and belongs to `Label_1`,

$$Charges = -3717 + 1387.91(1) + 273.81(50) + 32640.09(1) = 44,001.5$$  

* The **residual plot** indicates the variance in the estimations for individuals in `Label_3` is smaller than those in `Label_1` and `Label_2`.
* Generally, the greater the medical cost, the greater the error of the model.
* However, the model still performs well without under- and over-fitting the training data.
