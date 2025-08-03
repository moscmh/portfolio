import streamlit as st

st.set_page_config(page_title=None, 
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="expanded", 
                   menu_items=None)

st.title("British Airways Customer Booking Prediction")

# Sidebar
st.sidebar.header("Content")
button_intro = st.sidebar.button("Introduction")
button_model = st.sidebar.button("Model Training and Result")

buttons = [button_intro, button_model]

# Default Page
if True not in buttons:
    st.header("Introduction")
    st.image("https://banner2.cleanpng.com/20181113/iai/kisspng-british-airways-logo-oneworld-united-kingdom-qanta-wrl-ll-qts-1713923070015.webp", caption="photo from Wikipedia")
    st.markdown("&emsp;The goal of this project is to build a predictive model that can estimate the likelihood of a customer booking a flight with British Airways. The available dataset contains 13 features, including `number of passengers`, `sales channel`, `trip type`, etc.") 
    st.markdown("&emsp;The binary classes, complete or incomplete booking, are significantly imbalanced. To address this, downsampling on the majority class was applied before exploratory data analysis and preprocessing.")
    st.markdown("&emsp;A gradient boosting classifier was trained and optimsed using Optuna, achieving accuracy, recall, and F1 scores of around 65%. Future work may include data collection of the minority class, a new feature that records the month of the flight, and a focus on route preferences.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Preprocessing")
        st.markdown("OneHotEncoder, StandardScaler, OrdinalEncoder, ColumnTransformer, Pipeline from Scikit-Learn")
        st.markdown("### Model Building & Evaluation")
        st.markdown("GradientBoostingClassifier from Scikit-Learn, Optuna")

# Introduction
if button_intro:
    st.header("Introduction")
    st.image("https://www.google.com/imgres?q=british%20airways&imgurl=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fsco%2Fthumb%2F4%2F42%2FBritish_Airways_Logo.svg%2F1200px-British_Airways_Logo.svg.png&imgrefurl=https%3A%2F%2Fsco.wikipedia.org%2Fwiki%2FFile%3ABritish_Airways_Logo.svg&docid=9f8_A-xCbWNK3M&tbnid=TnjOZTQ9KY2crM&vet=12ahUKEwiBs5Smg--OAxX5rlYBHaknI6sQM3oECCIQAA..i&w=1200&h=188&hcb=2&ved=2ahUKEwiBs5Smg--OAxX5rlYBHaknI6sQM3oECCIQAA", caption="photo from Wikipedia")
    st.markdown("&emsp;The goal of this project is to build a predictive model that can estimate the likelihood of a customer booking a flight with British Airways. The available dataset contains 13 features, including `number of passengers`, `sales channel`, `trip type`, etc.") 
    st.markdown("&emsp;The binary classes, complete or incomplete booking, are significantly imbalanced. To address this, downsampling on the majority class was applied before exploratory data analysis and preprocessing.")
    st.markdown("&emsp;A gradient boosting classifier was trained and optimsed using Optuna, achieving accuracy, recall, and F1 scores of around 65%. Future work may include data collection of the minority class, a new feature that records the month of the flight, and a focus on route preferences.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("numpy, pandas")
        st.markdown("### Visualization Tools")
        st.markdown("matplotlib, seaborn")
        st.markdown("### Preprocessing")
        st.markdown("OneHotEncoder, StandardScaler, OrdinalEncoder, ColumnTransformer, Pipeline from Scikit-Learn")
        st.markdown("### Model Building & Evaluation")
        st.markdown("GradientBoostingClassifier from Scikit-Learn, Optuna")

# Model Training and Result
if button_model:
    st.header("Model Training and Result")
    st.markdown("![Confusion Matrix](https://github.com/moscmh/portfolio/blob/main/BritishAirways/plot/cm.png?raw=true)")
    st.markdown("* The model was optimsed using Optuna and all accuracy, recall, and F1 score are about 65%.")
    st.markdown("* The downsampled dataset enhanced the overall performance on both classes. Without downsampling, the model would easily score >80% accuracy, which simply predicts most of the instances as the majority class: incomplete booking.")
    st.markdown("![Feature Importance](https://github.com/moscmh/portfolio/blob/main/BritishAirways/plot/features.png?raw=true)")
    st.markdown("* The most important features are `purchase lead`, `length of stay`, `flight duration`, etc.")
    st.markdown("* Marketing may pay attention to how to increase the chance of booking completion for larger `purchase lead` and `length of stay`.")
