# Diabetic Retinopathy Classification using PCA-involved CNN models

Diabetic retinopathy (DR) is the leading cause of preventable blindness. With the development of medical imaging techniques and pattern recognition techniques, convolutional neural networks are helping doctors to identify the stage of DR patients faster. <br />

In this project, we explored various CNN models on classifying DR images and applied the PCA method on activation maps of these models to expedite the process (method proposed by [Garg et al.](https://arxiv.org/abs/1812.06224)). <br />

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Contributors](#contributors)
* [Data](#data)
* [Contents](#contents)
* [Shallow CNN](#shallow-cnn)
* [Deep CNN](#deep-cnn-models)

## Contributors
Jiyu Wang <br />
Chunlei Zhou <br />

## Contents
Folders and files in this repository:
* Data: the dataset of preprocessed images. Within /train and /test, images are stored in folders named with their labels, which enables the use of image data generator in Tensorflow.
* report_images: images created during analyses and used in reports
* Saved_files
    - history: training history of each model
    - models: saved files of trained models
    - PCA: analysis result of PCA on base models
    - training_times.csv: training time of each model
* sources
    - ShallowNet.py: functions to create ShallowNets
    - util.py: functions used in analysis
* 001-preprocessing.ipynb: notebook for image preprocessing
* 002 files: training and analyses on shallow CNN models
* 003 files: training and analyses on conventional CNN models using Tensorflow
    
<!-- Data -->
## Data
The DR image dataset we are using is from a kaggle competition ([APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)). Since the labels of the original test set was not provided, the original training set was divided into the current training set and test set.<br />

To remove the variance introduced by unrelated factors such as lighting condition when image was taken, the camera condition, etc., the DR images were preprocessed while critical information for DR classifying was maintained. The details of the prepocessing can be found in [001-preprocessing.ipynb](001-preprocessing.ipynb)

## Shallow CNN
Shallow CNN models implemented using Tensorflow would be a good tool to study the method of PCA-involved CNN models, due to our limited size of the training set and faster training time of shallow networks.<br />

* Shallow CNN base model

The **base shallow model** consists of four convolutional blocks, each of which contain two convolutional layers as well as a maxpooling layer and a batch normalization layer.

<p align="center">
  <img align="center" src="report_images/shallow_base.png" alt="base_model" width="600"/>
</p>
<div align="center"><b>Structure of the ShallowNet Base model</b></div><br />


* PCA analysis on base model

After the pretraining of the base model, **PCA analysis** was performed on its activation map and determined the critical number of filters/nodes of the conv layers and dense layers. The number of filters/nodes of conv layers and fc layers was determined by the number of critical components of each layer, while conv layers that contain fewer critical components than the layer ahead were also removed.

Below are the details of the shallow models:

<div align="center"><b>Comparisons on structures of ShallowNets</b></div>
<p align="center">
  <img align="center" src="report_images/shallow_models.png" alt="shallow_models" width="600"/>
</p>
<br />


* Performance of variant models

Three **variant models** were created based on the PCA result:

    - var1: only modify conv layers
    - var2: only modify fc layers
    - var3: modify both conv layers and fc layers

Performance of these three models on classification AUC is comparable to the base model, while training time and number of parameters are reduced.

<p align="center">
  <img align="center" src="report_images/metrics_comparison.png" alt="model_compare"/>
</p>
<div align="center"><b>Compare performance of ShallowNets</b></div><br />

Details of the analysis can be found in [this notebook](002c-shallow%20CNN_analysis.ipynb)

## Deep CNN models
Several deep CNN models, including DenseNet169, InceptionV3, MobileNetV2, NASNetMobile, ResNet152, VGG16, and Xception, are implemented.
<br/> We trained various models to explore the effect of PCA on different CNN models.
<br/> One thing that needs to be mentioned, we used the pre-trained model (weights = 'imagenet'). The **PCA analysis** was performed only on the fc layers.
* Performance of Original Models and Post-PCA Models
<br> 
Except for the DenseNet169, NASNetMobile, and ResNet152, all other models have similar model performance before and after PCA implementation.
<br/> Overall, the training time of each post_PCA model is slightly shorter than its corresponding original model.

<div align="center"><b>Compare AUC Scores of Deep CNN Models</b></div><br />
<p align="center">
  <img align="center" src="report_images/deep_cnn_model_metrics.png" alt="model_compare"/>
</p>

<div align="center"><b>Compare Training Time of Deep CNN Models</b></div><br />
<p align="center">
  <img align="center" src="report_images/deep_cnn_traintime_paranum.png" alt="model_compare"/>
</p>

Details of the analysis can be found in [this notebook](003b_Model_Evaluation.ipynb)