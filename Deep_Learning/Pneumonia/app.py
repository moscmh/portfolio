import streamlit as st

st.set_page_config(page_title=None, 
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="expanded", 
                   menu_items=None)

st.title("Pneumonia Detection Model")

# Sidebar
st.sidebar.header("Content")
button_intro = st.sidebar.button("Introduction")
button_eda = st.sidebar.button("Data Exploration")
button_model = st.sidebar.button("Model Training and Evaluation")

buttons = [button_intro, button_eda, button_model]

# Default Page
if True not in buttons:
    st.header("Introduction")
    st.image("https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/header.jpg?raw=true", caption="photo from Medscape")
    st.markdown("&emsp;This project builds deep learning models that detects pneumonia using X-ray images.") 
    st.markdown("&emsp;Two `CNN` models and a `Transfer Learning` model were built preliminarily. The simplest `CNN` model outperformed the `ResNet101V2` pre-trained model. The accuracies on `test set` were around `85%` and `67%` respectively.")
    st.markdown("&emsp;Further tuning and regularisation techniques need to be considered in order to improve the models. A model with `99%` accuracy is expected because pneumonia is a serious medical condition with a `concerning mortality rate`.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("random, numpy")
        st.markdown("### Visualisation")
        st.markdown("matplotlib")
        st.markdown("### Deep Learning Models")
        st.markdown("tensorflow")

# Introduction
if button_intro:
    st.header("Introduction")
    st.image("https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/header.jpg?raw=true", caption="photo from Medscape")
    st.markdown("&emsp;This project builds deep learning models that detects pneumonia using X-ray images.") 
    st.markdown("&emsp;Two `CNN` models and a `Transfer Learning` model were built preliminarily. The simplest `CNN` model outperformed the `ResNet101V2` pre-trained model. The accuracies on `test set` were around `85%` and `67%` respectively.")
    st.markdown("&emsp;Further tuning and regularisation techniques need to be considered in order to improve the models. A model with `99%` accuracy is expected because pneumonia is a serious medical condition with a `concerning mortality rate`.")

    with st.expander("Python Libraries", False):
        st.markdown("### Data Manipulations")
        st.markdown("random, numpy")
        st.markdown("### Visualisation")
        st.markdown("matplotlib")
        st.markdown("### Deep Learning Models")
        st.markdown("tensorflow")

# Data Exploration
if button_eda:
    st.header("Data Exploration")
    st.image("https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/images.png?raw=true", caption="Samples")
    st.markdown("* If data augmentations are required, flipping horizontally may yield acceptable results. However, it is not recommended as most of the images above show that the position of the hearts is more to the left side of the children. By flipping horizontally, the heart will be more to the right instead which is not common in practice. Therefore, training on those augmented images may not be helpful.")
    st.markdown("* As a result, rotation may be a better option in case of needing a data augmentation.")

# Model Training and Evaluation
if button_model:
    st.header("Model Training and Evaluation")
    st.markdown("## Simple CNN Model")
    st.markdown("&emsp;A simple CNN model which consists of a `Conv2D` layer, `MaxPooling2D` layer, and a `fully connected` layer. The accuracies on training and validation sets are both above `93%` at epoch `5`.")
    st.markdown("## ResNet101v2")
    st.markdown("&emsp;A pretrained `ResNet101v2` was implemented. Its training and validation accuracies were around `83%`.")
    st.image("https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/resnet101v2.png?raw=true", caption="ResNet101v2 Performance")

    st.markdown("## Evaluation")
    st.markdown("&emsp;Evaluation on the `simple CNN model` and the `ResNet101v2` model revealed that the former achieved a higher accuracy of `85%` than the pretrained model with `67%` accuracy.")