#!/usr/bin/env python
# coding: utf-8

"""
DR_classifier
used to organize the CNN models and PCA analysis
attributes:
    name - classifier name
    base - base model of the classifier, usually the default full model prior to PCA
    train - path to train dataset
    test - path to test dataset
functions:
    info - return list of current models
    load_new_model - load a

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


class DR_classifier:
    def __init__(self, name, base, dataset = "Data\\preprocessed_images", save_dir = "Saved_models"):
        self.name = name
        self.models = {'base': base}
        self.var_num = 0
        self.train_dir = os.path.join(dataset, "train")
        self.test_dir = os.path.join(dataset, "test")
        self.saves = os.path.join(save_dir, name)
        if not os.path.exists(self.saves):
            os.makedirs(self.saves)
        
    # check models
    def info(self):
        print("Current "+self.name +" models:")
        print(str(list(self.models.keys())))
    
    # load a trained model
    def load_new_model(self, model_file):
        c_name, m_name, _ = model_file.split('-')
        # check if object name matches
        if c_name != self.name:
            raise ValueError('Object name does not match')
        
        if m_name in list(self.models.keys()):
            print(m_name+" already existed. Replaced with the new model.")
            self.models.pop(m_name)
          
        model = tf.keras.models.load_model(model_file)
        self.models[m_name] = model
        print(m_name+' model loaded')
    
    # add variant models
    def add_models(self, mode = 'top', base_model = None, filter_num = 4096, dense_num = 2, class_num = 5):
        """
        adding variant models to the object
        Input:
            mode: 'top' - using base in the object, only changing the top layers
                'full' - using provided base. base must be provided
            base: base model in full mode
            filter_num: int or list of ints indicating number of nodes in each layer
            dense_num: number of dense layers. If filter_num is list, the length needs to match the dense_num
            class_num: number of classes
        
        """
        if mode == 'full':
            base = base_model
        elif mode == 'top':
            base = self.models['base']
        else:
            raise ValueError('Wrong mode')
        
        if type(filter_num) == int:
            filter_num = [filter_num]*dense_num
        elif type(filter_num) == list:
            if len(filter_num) != dense_num:
                raise ValueError('Number of items in filter_num does not match with dense_num')
        else:
            raise ValueError('Wrong input type of filter_num')


        x = layers.Flatten(name='flatten')(base.output)

        for i in range(dense_num):
            name = 'top_fc'+str(i+1)
            x = layers.Dense(filter_num[i], activation='relu', name=name)(x)

        pred = layers.Dense(class_num, activation = 'softmax', name='predictions')(x)
        model = keras.Model(base.input, pred)
        
        # save model in models
        if 'ori' not in list(self.models.keys()):
            self.models['ori'] = model
            print('ori model created')
            print('Current models: '+str(list(self.models.keys())))
        else:
            m_name = 'var'+str(self.var_num+1)
            self.var_num += 1
            self.models[m_name] = model
            print(m_name+' model created')
            print('Current models: '+str(list(self.models.keys())))
    
    # train a model
    def train(self, model_name, dense_num = 2, epoch_num = 50):
        '''
        train a model
        Input:
            model_name: key of the models dict
            dense_num: number of dense layers added to the top
            epoch_num: number of epochs to train
        Output:
            csv file with training history
            h5 file with trained model
            txt file with training time
        '''
        if model_name not in self.models.keys():
            raise ValueError('Model not existed')
        elif model_name == 'base':
            raise ValueError('Base model can not be trained')
        model = self.models[model_name]
        
        # create data generators
        train_datagen = ImageDataGenerator(rescale=1./255, brightness_range = [0.5, 1.0], horizontal_flip = True, vertical_flip = True)
        train_img = train_datagen.flow_from_directory(directory = self.train_dir, target_size=(224, 224), color_mode="rgb",
                                                                batch_size=64, class_mode="categorical", shuffle=True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_img = test_datagen.flow_from_directory(directory = self.test_dir, target_size=(224, 224), color_mode="rgb", 
                                                              batch_size=64, class_mode="categorical", shuffle=True)
        
        # metrics
        metrics = [keras.metrics.CategoricalAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]
        
        # weights of each class
        weights = {0: 0.4062717770034843, 1: 1.976271186440678,2: 0.7333333333333333,3: 3.785714285714286,4: 2.4703389830508473}
        
        # get index of the first fc layer of the top layers
        idx = -(dense_num+1)
        for layer in model.layers[:idx]:
            layer.trainable=False
        for layer in model.layers[idx:]:
            layer.trainable=True
            
        model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=metrics)
        
        start = time.time()
        model_history = model.fit(train_img, epochs=epoch_num, steps_per_epoch = 46, validation_data=test_img, validation_steps = 11, class_weight = weights, verbose = 1)
        finish = time.time()
        
        df = pd.DataFrame(model_history.history)
        
        # output
        hist_name = self.name + '-' + model_name + '-train_history.csv'
        save_name = self.name + '-' + model_name + '-model.h5'
        time_name = self.name + '-' + model_name + '-time.txt'
        
        df.to_csv(hist_name) #.csv file
        model.save(save_name) #.h5 file
        with open(time_name,'w') as f: # .txt file
            f.write('%.4f' % (finish - start))
            
        print('Files created:')
        print([hist_name, save_name, time_name])
    
    # PCA analysis on a model
    def model_PCA(self, model_name, mode = 1):
        '''
        analyze layer activations of the model
        Input:
            model_name: key of the models dict
            mode: 1 - analyze top-fc only; 2 - analyze conv and fc
        '''
        if model_name not in self.models.keys():
            raise ValueError('Model not existed')
            
        if mode == 1:
            target = ['top']
            mini_filter = {'top_fc':{}}
        elif mode == 2:
            target = ['conv', 'top']
            mini_filter = {'top_fc':{}, 'conv':{}}
        else:
            raise ValueError('mode can only be 1 or 2')

        model = self.models[model_name]
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_img = test_datagen.flow_from_directory(directory = self.test_dir, target_size=(224, 224), color_mode="rgb", 
                                                              batch_size=64, class_mode="categorical", shuffle=True)
        
        # obtain activation map
        get_layer_output = K.function(inputs = model.input, outputs = {layer.name:layer.output for layer in model.layers})
        layer_output = get_layer_output(test_img[0])
        
        def analyze_PCA(output):
            if output.ndim == 4:
                n, h, w, m = output.shape
                output = output.reshape((-1,m))
            pca= PCA()
            pca.fit(output)
            return pca.explained_variance_ratio_
        def layer_check(layer):
            for name in target:
                if name in layer:
                    return True
            return False
        
        cnt = 1
        for key, val in layer_output.items():
            if layer_check(key):
                variance = analyze_PCA(val)
                cumVar = np.cumsum(variance)
                mini = np.argmax(cumVar[cumVar<0.999])+2
                l_name = str(cnt)+'-'+key
                if 'top' in key:
                    mini_filter['top_fc'][l_name] = mini
                elif 'conv' in key:
                    mini_filter['conv'][l_name] = mini
                cnt += 1
        
        mini_name = self.name + '-' + model_name +'-postPCA.txt'
        with open(mini_name,'w') as f: # .txt file
            f.write("=== number of critical filters in each layer ===\n")
            for key, val in mini_filter.items():
                f.write("##"+key+" layers:\n")
                for k, v in val.items():
                    f.write(str(k)+": "+str(v)+";")
                f.write("\n List form: ")
                f.write(str(val.values()))
                f.write("\n\n")
        
        print("PCA on "+model_name+" finished")
        print(mini_name+" created")
        print("filter numbers should be: ")
        for key, val in mini_filter.items():
            print(key+":")
            print(str(val.values()))

