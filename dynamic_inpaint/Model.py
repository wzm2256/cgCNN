import keras
import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.engine.topology import Layer

from keras.layers import Conv2D, Conv3D, AveragePooling2D, AveragePooling3D, Deconv2D, UpSampling2D, pooling, BatchNormalization, Activation, concatenate, SeparableConv2D, ThresholdedReLU

class small_1():
    """
    with large
    """
    
    def __init__(self):
        self.Layer_List = list()
        self.Layer_List.append(Conv3D(32, (3, 3, 3), padding='same', name='conv1'))
        self.Layer_List.append(BatchNormalization(name='batch_norm1'))
        self.Layer_List.append(Activation("relu", name='relu1'))

        self.Layer_List.append(Conv3D(32, (3, 3, 3), padding='same', name='conv2'))
        self.Layer_List.append(BatchNormalization(name='batch_norm2'))
        self.Layer_List.append(Activation("relu", name='relu2'))

        self.Layer_List.append(AveragePooling3D(2, strides=2, name='block1_pool'))

        self.Layer_List.append(Conv3D(32, (3, 3, 3), padding='same', name='conv3'))
        self.Layer_List.append(BatchNormalization(name='batch_norm3'))
        self.Layer_List.append(Activation("relu", name='relu3'))

    def run(self, x):
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