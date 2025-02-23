# Pneumonia Detection Model
&emsp;This project builds deep learning models that detects pneumonia using X-ray images.

&emsp;Two `CNN` models and a `Transfer Learning` model were built preliminarily. The simplest `CNN` model outperformed the `ResNet101V2` pre-trained model. The accuracies on `test set` were around `85%` and `67%` respectively.

&emsp;Further tuning and regularisation techniques need to be considered in order to improve the models. A model with `99%` accuracy is expected because pneumonia is a serious medical condition with a `concerning mortality rate`.

## Python Libraries
### Data Manipulations
random, numpy
### Visualisation
matplotlib
### Deep Learning Models
tensorflow

# Data Exploration
![X-Ray Images](https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/images.png?raw=true)
* If data augmentations are required, flipping horizontally may yield acceptable results. However, it is not recommended as most of the images above show that the position of the hearts is more to the left side of the children. By flipping horizontally, the heart will be more to the right instead which is not common in practice. Therefore, training on those augmented images may not be helpful.
* As a result, rotation may be a better option in case of needing a data augmentation.

# Modelling
## Simple CNN Model
&emsp;A simple CNN model which consists of a `Conv2D` layer, `MaxPooling2D` layer, and a `fully connected` layer. The accuracies on training and validation sets are both above `93%` at epoch `5`.
## ResNet101v2
&emsp;A pretrained `ResNet101v2` was implemented. Its training and validation accuracies were around `83%`.
![ResNet101v2](https://github.com/moscmh/portfolio/blob/main/Deep_Learning/Pneumonia/plot/resnet101v2.png?raw=true)

# Evaluation
&emsp;Evaluation on the `simple CNN model` and the `ResNet101v2` model revealed that the former achieved a higher accuracy of `85%` than the pretrained model with `67%` accuracy.