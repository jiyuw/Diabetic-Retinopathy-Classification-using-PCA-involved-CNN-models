#!/usr/bin/env python
# coding: utf-8

"""
model naming convention: <CNN type>_<version num>
files saved:
    model: .h5 files in saved_files/model/
    history: .csv files in saved_files/history/
    time: .csv file - saved_files/training_times.csv
Functions in analysis:
    load_model: load a saved model
    train_model: train a model and save outputs
    model_PCA: analyze a trained model with PCA
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os


# load a trained model
def load_model(model_file, model_dir=os.path.join("Saved_files", "models")):
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        raise ValueError(f'Model {model_file} not found')

    model_name, _ = model_file.split('.')

    model = tf.keras.models.load_model(model_path)
    print(model_name + ' model loaded')
    return model


# train a model
def train_model(model, model_name, dense_num=2, epoch_num=50,
                train_dir=os.path.join('Data', 'preprocessed_images', 'train'),
                test_dir=os.path.join('Data', 'preprocessed_images', 'test'),
                save_dir='Saved_files'):
    """
    train a model
    Input:
        model: model to be trained
        model_name: name of the model for file saving
        dense_num: number of dense layers added to the top
        epoch_num: number of epochs to train
        train_dir: path to training set
        test_dir: path to test set
    Output:
        csv file with training history
        h5 file with trained model
        update csv file to save traning time
    """

    # create data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255, brightness_range=[0.5, 1.0], horizontal_flip=True,
                                       vertical_flip=True)
    train_img = train_datagen.flow_from_directory(directory=train_dir, target_size=(224, 224), color_mode="rgb",
                                                  batch_size=64, class_mode="categorical", shuffle=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_img = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224), color_mode="rgb",
                                                batch_size=64, class_mode="categorical", shuffle=True)

    # metrics
    metrics = [keras.metrics.CategoricalAccuracy(name='accuracy'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]

    # weights of each class
    weights = {0: 0.4062717770034843, 1: 1.976271186440678, 2: 0.7333333333333333, 3: 3.785714285714286,
               4: 2.4703389830508473}

    # get index of the first fc layer of the top layers
    idx = -(dense_num + 1)
    for layer in model.layers[:idx]:
        layer.trainable = False
    for layer in model.layers[idx:]:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=metrics)

    start = time.time()
    model_history = model.fit(train_img, epochs=epoch_num, steps_per_epoch=46, validation_data=test_img,
                              validation_steps=11, class_weight=weights, verbose=1)
    finish = time.time()

    print(f"{model_name} model trained")

    df = pd.DataFrame(model_history.history)

    # output
    hist_name = os.path.join(save_dir, model_name+'-train_history.csv')
    save_name = os.path.join(save_dir, model_name + '-model.h5')
    df.to_csv(hist_name)  # .csv file
    model.save(save_name)  # .h5 file

    # save training time
    time_file = os.path.join(save_dir, 'training_times.csv')
    if os.path.exists(time_file):
        time_df = pd.read_csv(time_file)
    else:
        time_df = pd.DataFrame(columns=['model', 'training time', 'epochs'])

    new_model = {'model': model_name, 'training time': '%.4f' % (finish - start), 'epochs': 50}
    time_df = time_df.append(new_model, ignore_index=True).sort_values(by=['model'])
    time_df.to_csv(time_file, index=False)

    print('Files created:')
    print([hist_name, save_name, time_file])

    return model


# PCA analysis on a model
def model_PCA(model, model_name, mode=1, test_dir=os.path.join('Data', 'preprocessed_images', 'test'),
              save_dir='Saved_files'):
    """
    analyze layer activations of the model
    Input:
        model: model to analyze
        model_name: name of the model
        mode: 1 - analyze top-fc only; 2 - analyze conv and fc
    """

    if mode == 1:
        target = ['top']
    elif mode == 2:
        target = ['conv', 'top']
    else:
        raise ValueError('mode can only be 1 or 2')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_img = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224), color_mode="rgb",
                                                batch_size=64, class_mode="categorical", shuffle=True)

    # obtain activation map
    get_layer_output = K.function(inputs=model.input, outputs={layer.name: layer.output for layer in model.layers})
    layer_output = get_layer_output(test_img[0])

    def analyze_PCA(output):
        if output.ndim == 4:
            n, h, w, m = output.shape
            output = output.reshape((-1, m))
        pca = PCA()
        pca.fit(output)
        return pca.explained_variance_ratio_

    def layer_check(layer):
        for name in target:
            if name in layer:
                return name
        return ''

    output = pd.DataFrame(columns=['model', 'layer_cat', 'layer', '#pre-PCA', '#post-PCA'])

    cnt = 1
    for key, val in layer_output.items():
        cat = layer_check(key)
        if not cat:
            variance = analyze_PCA(val)
            cumVar = np.cumsum(variance)
            mini = np.argmax(cumVar[cumVar < 0.999]) + 2
            curr_row = {'model': model_name, 'layer_cat': cat, 'layer': key, '#pre-PCA': val.shape[1], '#post-PCA': mini}
            output = output.append(curr_row, ignore_index=True).sort_values(by=['layer_cat', 'layer'])

    PCA_file = os.path.join(save_dir, model_name+'-postPCA.csv')
    output.to_csv(PCA_file, index=False)

    print("PCA on " + model_name + " finished")
    print(PCA_file + " created")
