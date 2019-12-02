import keras
import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.engine.topology import Layer

from keras.layers import Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, Deconv2D, UpSampling2D, pooling, BatchNormalization, Activation, concatenate, SeparableConv2D
from keras.layers.advanced_activations import ThresholdedReLU

class Sound_inpaint():
    
    def __init__(self):
        self.Layer_List = list()

        self.Layer_List.append(Conv1D(128, 25, strides=3, padding='valid', name='conv1'))
        self.Layer_List.append(BatchNormalization(name='batch_norm1'))
        self.Layer_List.append(Activation("relu", name='relu1'))

        self.Layer_List.append(Conv1D(128, 25, strides=5, padding='valid', name='conv2'))
        self.Layer_List.append(BatchNormalization(name='batch_norm2'))
        self.Layer_List.append(Activation("relu", name='relu2'))

        self.Layer_List.append(Conv1D(128, 25, strides=10, padding='valid', name='conv3'))
        self.Layer_List.append(BatchNormalization(name='batch_norm3'))
        self.Layer_List.append(Activation("relu", name='relu3'))

    def run(self, x):
        # Return the var list
        self.hid_dict = dict()

        for i in range(len(self.Layer_List)):
            x = self.Layer_List[i](x)
            if 'relu' in self.Layer_List[i].name:
                x = tf.clip_by_value(x, 0, 1)

            self.hid_dict[self.Layer_List[i].name] = x
        self.out = x

        self.var_list = list()
        for i in range(len(self.Layer_List)):
            tmp = self.Layer_List[i].trainable_weights
            for j in range(len(tmp)):
                self.var_list.append(tmp[j])

        return self.out, self.hid_dict, self.var_list
