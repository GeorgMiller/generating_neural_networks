import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.optimizers import RMSprop
from keras import Model


def Actor(input_shape, output_shape, lr):
        
    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(output_shape, activation="softmax", kernel_initializer='he_uniform')(X)


    Actor = Model(inputs = X_input, outputs = action)
    #Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    return Actor

def Critic(input_shape, output_shape, lr):


    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    value = Dense(1, kernel_initializer='he_uniform')(X)

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Critic

network = Actor((80,80,1),3, 0.01)
network.summary()
print(network.layers[2].trainable_variables)


class Networks():

    def __init__(self,input_shape, output_shape, lr):


        self.input = Input(input_shape)

        #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
        #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
        #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
        self.flatten = Flatten(input_shape=input_shape)

        self.dense = Dense(512, activation="elu", kernel_initializer='he_uniform')
        #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
        #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

        self.output = Dense(output_shape, activation="softmax", kernel_initializer='he_uniform')
        

        Actor = Model(inputs = X_input, outputs = action)
        Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))
    
    def call(self):
        
        return output



        

class Critic(keras.Model):

    def __init__(self, input_shape, output_shape, lr):

        self.input = Input(input_shape)

        #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
        #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
        #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
        self.flatten = Flatten(input_shape=input_shape)(X_input)

        self.dense = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
        #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
        #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

        self.value = Dense(1, kernel_initializer='he_uniform')(X)

        Critic = Model(inputs = X_input, outputs = value)
        Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    def call(self):

        return output

class Hypernetwork(keras.Model):

    def __init__(self, target_network, name):

        if name == 'bayesian':

            

            #this is the main structure which is more or less indepente

            self.dense_1 = Dense(300, activation="relu")
            self.dense_2 = Dense(855, activation="relu")

            #these are the distributed sublayers

            self.w1_dense_1 = Dense(40, activation="relu")
            self.w1_dense_2 = Dense(26, activation="relu") 
            
            self.w2_dense_1 = Dense(100, activation="relu")
            self.w2_dense_2 = Dense(801, activation="relu")

            self.w3_dense_1 = Dense(100, activation="relu")
            self.w3_dense_2 = Dense(785, activation="relu")

            self.w4_dense_1 = Dense(60, activation="relu")
            self.w4_dense_2 = Dense(90, activation="relu")


    def call(self, inputs):

        index_1 = 32*15
        index_2 = index_1 + 16*15
        index_3 = index_2 + 8*15
        index_4 = index_3 + 1*15

        layer_1 = 16
        layer_2 = 32
        layer_3 = 254
        layer_4 = 3

        output = []

        x = self.dense_1(inputs)
        x = self.dense_2(x)
        #print(x.shape)

        input_w1 = x[:,:index_1]
        #print("input_w1", input_w1)
        input_w1 = tf.reshape(input_w1,(32,-1))


        for step in range(layer_1):

            w1 = input_w1[step,:]
            w1 = tf.reshape(w1,(-1,1))

            w1 = self.w1_dense_1(w1)
            w1 = self.w1_dense_2(w1)

            output = tf.concat([output,w1[1]], 0)
            
        input_w2 = x[:,index_1:index_2]
        #print("input_w2", input_w2)
        input_w2 = tf.reshape(input_w1,(16,-1))
        
        for step in range(layer_2):
           
            w2 = input_w2[step,:]
            w2 = tf.reshape(w2,(-1,1))
            w2 = self.w2_dense_1(w2)
            w2 = self.w2_dense_2(w2)
            output = tf.concat([output,w2[1]], 0)

        input_w3 = x[:,index_2:index_3]
        #print("input_w3", input_w3)
        input_w3 = tf.reshape(input_w3,(8,-1))

        for step in range(layer_3):

            w3 = input_w3[step,:]
            w3 = tf.reshape(w3,(-1,1))
            w3 = self.w3_dense_1(w3)
            w3 = self.w3_dense_2(w3)
            output = tf.concat([output,w3[1]], 0)

        input_w4 = x[:,index_3:index_4]
        #print("input_w4", input_w4)
        input_w4 = tf.reshape(input_w4,(1,-1))

        for step in range(layer_4):
            
            w4 = input_w4[step,:]
            w4 = tf.reshape(w4,(-1,1))
            w4 = self.w4_dense_1(w4)
            w4 = self.w4_dense_2(w4)
            output = tf.concat([output,w4[1]], 0)

        #z = np.random.uniform(low = -1, high = 1, size = 5070)
        #output = tf.concat([output,z],0)

        return output
