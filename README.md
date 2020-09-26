# Diabetic Retinopathy Classification using PCA-involved CNN models

Diabetic retinopathy (DR) is the leading cause of preventable blindness. With the development of medical imaging techniques and pattern recognition techniques, convolutional neural networks are helping doctors to identify the stage of DR patients faster. <br />

In this project, we explored various CNN models on classifying DR images and applied the PCA method on activation maps of these models to expedite the process (method proposed by [Garg et al.](https://arxiv.org/abs/1812.06224)). <br />

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Contributors](#contributors)
* [Data](#data)
* [Contents](#contents)
* [Shallow CNN](#shallow-cnn)

## Contributors
Jiyu Wang <br />
Chunlei Zhou <br />

## Contents


<!-- Data -->
## Data
The DR image dataset we are using is from a kaggle competition ([APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)). Since the labels of the original test set was not provided, the original training set was divided into the current training set and test set.<br />

To remove the variance introduced by unrelated factors such as lighting condition when image was taken, the camera condition, etc., the DR images were preprocessed while critical information for DR classifying was maintained. The details of the prepocessing can be found in [001-preprocessing.ipynb](001-preprocessing.ipynb)

## Shallow CNN
Shallow CNN models implemented de novo using Tensorflow would be a good tool to study the method of PCA-involved CNN models, due to our limited size of the training set and faster training time of shallow networks.<br />

The base 
