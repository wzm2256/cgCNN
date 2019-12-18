import keras
import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.engine.topology import Layer

from keras.layers import Conv2D, AveragePooling2D, Deconv2D, UpSampling2D, pooling, BatchNormalization, Activation, concatenate, SeparableConv2D
from keras.layers.advanced_activations import ThresholdedReLU

class model1():

    
    def __init__(self, num_Com_layer):
        self.num_Com_layer = num_Com_layer
        self.Layer_List = list()
        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv1'))
        self.Layer_List.append(BatchNormalization(name='batch_norm1'))
        self.Layer_List.append(Activation("relu", name='relu1'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv2'))
        self.Layer_List.append(BatchNormalization(name='batch_norm2'))
        self.Layer_List.append(Activation("relu", name='relu2'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv3'))
        self.Layer_List.append(BatchNormalization(name='batch_norm3'))
        self.Layer_List.append(Activation("relu", name='relu3'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv4'))
        self.Layer_List.append(BatchNormalization(name='batch_norm4'))
        self.Layer_List.append(Activation("relu", name='relu4'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv5'))
        self.Layer_List.append(BatchNormalization(name='batch_norm5'))
        self.Layer_List.append(Activation("relu", name='relu5'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='valid', name='conv6'))
        self.Layer_List.append(BatchNormalization(name='batch_norm6'))
        self.Layer_List.append(Activation("relu", name='relu6'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        self.Composite_List = list()
        self.scale = np.array([15, 35, 55, 75, 135])
        n = 32
        for i in range(num_Com_layer):
            self.Composite_List.append(Conv2D(n, 15, padding='valid', dilation_rate=int(self.scale[i] / 15),  name='conv1'+str(i)))
            self.Composite_List.append(BatchNormalization(name='batch_Com' + str(i)))

    def run(self, x):
        self.hid_dict = dict()

        for i in range(self.num_Com_layer):
            tmp = self.Composite_List[i * 2](x)
            tmp = self.Composite_List[i * 2  + 1](tmp)
            self.hid_dict['composite' + str(i+1)] = tf.clip_by_value(tmp, 0, 1)

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

        for i in range(len(self.Composite_List)):
            tmp = self.Composite_List[i].trainable_weights
            for j in range(len(tmp)):
                self.var_list.append(tmp[j])

        return self.out, self.hid_dict, self.var_list, self.num_Com_layer
