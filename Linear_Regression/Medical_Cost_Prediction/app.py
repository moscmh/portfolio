import streamlit as st

st.set_page_config(page_title=None, 
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="expanded", 
                   menu_items=None)

st.title("Medical Cost Prediction")

# Sidebar
st.sidebar.header("Content")
button_intro = st.sidebar.button("Introduction")
button_eda = st.sidebar.button("Exploratory Data Analysis")
button_model = st.sidebar.button("Model Training and Result")

buttons = [button_intro, button_eda, button_model]

# Default Page
if True not in buttons:
    st.header("Introduction")
    st.image("https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/header.jpeg?raw=true", caption="photo from SBS News")
    st.markdown("&emsp;The [dataset](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) contains `1338` individuals' information with `6` attributes. They are `age`, `sex`, `BMI`, `number of children`, `whether is a smoker`, `residential region`. The outcome variable is the medical cost. This project showed a promising predictive **linear regression** model for estimating medical cost. It can be useful for insurance companies and government to determine policy cost and to allocate resources respectively.") 
    st.markdown("&emsp;Eventually, after conducting **exploratory data analysis** and **feature engineering**, a new feature was created that may be related to `hospital choice`. Together with `age` and `whether is a smoker`, a regression model with approximately 96.6% accuracy was built.")
    st.markdown("&emsp;Future work requires researching about the exact feature that was created. ")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Clustering")
        st.markdown("DBSCAN, GaussianMixture from Scikit-Learn")
        st.markdown("### Model Building & Evaluation")
        st.markdown("LinearRegression from Scikit-Learn")

# Introduction
if button_intro:
    st.header("Introduction")
    st.image("https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/header.jpeg?raw=true", caption="photo from SBS News")
    st.markdown("&emsp;The [dataset](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv) contains `1338` individuals' information with `6` attributes. They are `age`, `sex`, `BMI`, `number of children`, `whether is a smoker`, `residential region`. The outcome variable is the medical cost. This project showed a promising predictive **linear regression** model for estimating medical cost. It can be useful for insurance companies and government to determine policy cost and to allocate resources respectively.") 
    st.markdown("&emsp;Eventually, after conducting **exploratory data analysis** and **feature engineering**, a new feature was created that may be related to `hospital choice`. Together with `age` and `whether is a smoker`, a regression model with approximately 96.6% accuracy was built.")
    st.markdown("&emsp;Future work requires researching about the exact feature that was created. ")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Clustering")
        st.markdown("DBSCAN, GaussianMixture from Scikit-Learn")
        st.markdown("### Model Building & Evaluation")
        st.markdown("LinearRegression from Scikit-Learn")

# Exploratory Data Analysis
if button_eda:
    st.header("Exploratory Data Analysis")
    st.write("The important parts of the exploratory data analysis are listed in the following.")

    with st.expander("Categorical Features"):
        st.markdown("## Categorical Features")
        st.markdown("![Categorical Features](https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/plot/categorical.png?raw=true)")
        st.markdown("* `Sex` and `residential region` were found to have no significant relationship with `medical cost`.")
        st.markdown("* `Whether a smoker` showed a significant difference in medical cost between smokers and non-smokers.")

    with st.expander("Numeric Features"):
        st.markdown("## Numeric Features")
        st.markdown("![Numeric Features](https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/plot/numeric.png?raw=true)")
        st.markdown("* Three scatterplots were shown to examine the relationships of `Age` and `BMI` with `medical cost`.")
        st.markdown("* `BMI` did not show a linear relationship with `medical cost`.")
        st.markdown("* There were three trendlines in the scatterplot for `medical cost` vs `age`. Clustering was conducted next to efficiently label the three different groups.")

    with st.expander("Feature Engineering"):
        st.markdown("## Feature Engineering")
        st.markdown("![Feature engineering](https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/plot/feature_engineering.png?raw=true)")
        st.markdown("* **Guassian Mixture Model** was used to efficiently cluster the individuals from the three trendlines respectively.")
        st.markdown("* The trendlines could be explained by the choice of hospitals.")

# Model Training and Result
if button_model:
    st.header("Model Training and Result")
    st.markdown("![Metrics](https://github.com/moscmh/portfolio/blob/main/Linear_Regression/Medical_Cost_Prediction/plot/metrics.png?raw=true)")
    st.markdown("* Eventually, `age`, `BMI`, and the `newly created feature` were used to train a linear regression model.")
    st.markdown("* The accuracy of the model on training and testing data are `96.7%` and `96.6%` respectively.")
    st.markdown("* The model can be expressed by the following equation. `Label_1` and `Label_2` are binary values that belong to the categorical `newly created feature`. The sum of the binary values for this 2 labels does not exceed 1. If both are `0`, the individual has `Label_3` as the value of the `newly created feature`.")
    st.markdown("$$Charges = -3717.00 + 1387.91(Smoker) + 273.81(Age) + 32640.09(Label_1) + 13200.32(Label_2)$$  ")
    st.markdown("* For instance, if an individual is a `smoker`, `50` years old, and belongs to `Label_1`,")
    st.markdown("$$Charges = -3717 + 1387.91(1) + 273.81(50) + 32640.09(1) = 44,001.5$$  ")
    st.markdown("* The **residual plot** indicates the variance in the estimations for individuals in `Label_3` is smaller than those in `Label_1` and `Label_2`.")
    st.markdown("* Generally, the greater the medical cost, the greater the error of the model.")
    st.markdown("* However, the model still performs well without under- and over-fitting the training data.")