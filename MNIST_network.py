import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
#import hypernetwork


class MNIST_model(tf.keras.Model):

    def __init__(self):
        super(MNIST_model, self).__init__()

        self.conv2D_1 = Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(28,28,1))
        self.maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same")
        self.conv2D_2 = Conv2D(16, (5, 5), padding="same", activation="relu")
        self.maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = Dense(8, activation= "relu")
        self.dense_2 = Dense(10, activation= "softmax")

    def call(self, inputs):

        x = self.conv2D_1(inputs)
        x = self.maxpool_1(x)
        x = self.conv2D_2(x)
        x = self.maxpool_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x

