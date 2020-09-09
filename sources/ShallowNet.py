#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import pandas as pd
import numpy as np


# In[13]:


def ShallowNet(include_top=True, filter_map=None, n_class=5):
    """
    Create a ShallowNet model. The default model has 4 blocks with 2 conv layers in each block
    Arguments:
        include_top: whether to include the 3 fully-connected
          layers at the top of the network.
        filter_map: list of numbers indicating the number of filters in each layer; 0 means omitted layer
        n_class: number of classes to classify, only used when include_top = True
    """
    img_input = x = layers.Input(shape=(224, 224, 3))

    # set filter_map
    if filter_map is None:
        filter_map = [32, 32, 64, 64, 128, 128, 256, 256]
    elif len(filter_map) != 8:
        raise ValueError('Insufficient filter_map')

    # counter of blocks
    block = 1

    for b in range(4):
        block_name = 'block' + str(block)
        block_idx = [b * 2, b * 2 + 1]
        block_filters = [filter_map[i] for i in block_idx if filter_map[i] != 0]

        # if both layers in the same block have 0 filters, then skip this block
        if not block_filters:
            continue
        else:
            cnt = 1
            for num in block_filters:
                x = layers.Conv2D(num, (3, 3), activation='relu', padding='same',
                                  name=block_name + '_conv' + str(cnt))(x)
                cnt += 1

            x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=block_name + '_pool')(x)
            x = layers.BatchNormalization(name=block_name + '_batch')(x)
            block += 1

    # add top layers if include_top = True
    if include_top:
        # fc block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(100, activation='relu', name='fc1')(x)
        x = layers.Dense(100, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.25)(x)
        # output layer
        x = layers.Dense(n_class, activation='softmax',
                         name='predictions')(x)

    model = keras.Model(img_input, x)

    return model


# In[16]:


if __name__ == '__main__':
    base = ShallowNet(include_top=False)
    flatten = layers.Flatten(name='flatten')(base.output)
    fc1 = layers.Dense(4096, activation='relu', name='fc1')(flatten)
    fc2 = layers.Dense(4096, activation='relu', name='fc2')(fc1)
    pred = layers.Dense(5, activation='softmax', name='predictions')(fc2)
    model = keras.Model(base.input, pred)
    model.summary()
