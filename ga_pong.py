import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense



input_shape =(80,80,1)

states = keras.Input(input_shape, dtype='float32')
x = Conv2D(16, (8, 8), padding="same", activation="relu", dtype='float32')(states)
print(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(4,4), padding="same",dtype='float32')(x)
print(x)
x = Conv2D(32, (4, 4), padding="same", activation="relu", dtype='float32')(x)
print(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same",dtype='float32')(x)
print(x)
x = tf.keras.layers.Flatten()(x)
print(x)
x = Dense(254, activation= "relu",dtype='float32')(x)
print(x)


actions = Dense(3, activation= "softmax",dtype='float32')(x)
values = Dense(1,activation='linear',dtype='float32')(x)
    
network = tf.keras.Model(inputs = states, outputs = [actions, values])



def set_parameters(network, weights):

    weights_pred = weights_pred.astype('float32')
    index_w1 = 800
    index_b1 = 832

    index_w2 = index_b1 + 12800
    index_b2 = index_b1 + 12816

    index_w3 = index_b2 + 6272
    index_b3 = index_b2 + 6280

    index_w4 = index_b3 + 80
    index_b4 = index_b3 + 90

    kernel_1 = weights_pred[:index_w1]

    #print(kernel_1.dtype)
    kernel_1 = tf.reshape(kernel_1,(5,5,1,32))
    b_1 = weights_pred[index_w1:index_b1]
    
    kernel_2 = weights_pred[index_b1:index_w2]
    kernel_2 = tf.reshape(kernel_2,(5,5,32,16))
    b_2 = weights_pred[index_w2:index_b2]
    
    kernel_3 = weights_pred[index_b2:index_w3]
    kernel_3 = tf.reshape(kernel_3,(784,8))
    b_3 = weights_pred[index_w3:index_b3]
    
    kernel_4 = weights_pred[index_b3:index_w4]
    kernel_4 = tf.reshape(kernel_4,(8,10))
    b_4 = weights_pred[index_w4:index_b4]
    
    mnist_model.layers[0].kernel = kernel_1
    mnist_model.layers[0].bias = b_1

    mnist_model.layers[2].kernel = kernel_2
    mnist_model.layers[2].bias = b_2

    mnist_model.layers[5].kernel = kernel_3
    mnist_model.layers[5].bias = b_3

    mnist_model.layers[6].kernel = kernel_4
    mnist_model.layers[6].bias = b_4
    #print(mnist_model.layers[2].kernel)



train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3) 


batch_size = 32
epsilon = 0.5
workers = []
loss_dic = []
acc_dic = []
elite_workers = []

population_size = 50
weight_space = 20018

#create the randomized weight_space for each worker

for i in range(population_size):

    z = np.random.normal(loc=0, scale=0.01, size = weight_space)

    index_w1 = 800
    index_b1 = 832

    index_w2 = index_b1 + 12800
    index_b2 = index_b1 + 12816

    index_w3 = index_b2 + 6272
    index_b3 = index_b2 + 6280

    index_w4 = index_b3 + 80
    index_b4 = index_b3 + 90


    z[index_w1:index_b1] = np.zeros(32)

    z[index_w2:index_b2] = np.zeros(16)

    z[index_w3:index_b3] = np.zeros(8)

    z[index_w4:index_b4] = np.zeros(10)

    
    workers.append([z])    
    
    
for step in range(20000):
  idx = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
  x, y = x_train[idx], y_train[idx]
    
  with tf.GradientTape() as tape:

    for worker in workers: 
        
        set_parameters(network,worker[0])

        #now the worker should run for one epoch and return the rewards and the loss
        #the paper should be checked again in order to determine the used loss function
        rewards, loss = run_episode(network)

        loss_dic.append(loss)
        reward_dic.append(acc)
        
    acc_dic = np.array(acc_dic)
    elite_key_loss = np.argsort(acc_dic)[45:50]
    
    for key in elite_key_loss:
        elite_workers.append(workers[key])

    loss_dic = np.array(loss_dic)
    elite_key_loss = np.argsort(loss_dic)[:5]

    for key in elite_key_loss:
        elite_workers.append(workers[key])
    elite = workers[elite_key_loss[0]]
    

    #set the workers list empty
    workers = []

    #always take the one with the lowest loss value
    elite_acc = acc_dic[elite_key_loss[0]]
    elite_loss = loss_dic[elite_key_loss[0]]

    for i in range(population_size - 1):

        i = random.randint(0,9)
        a = epsilon * np.random.normal(0,1,size=20018)

        new_worker = elite_workers[i] + a
        workers.append(new_worker) 
    
    #set the elite_workers empty
    workers.append(elite)

    
    if step % 10 == 0:
        print('\n Step: {}, validation set accuracy: {:2.4f}, loss: {:2.4f}'.format(step, elite_acc, elite_loss))
        #print(preds)
        #print(elite_key_loss)
        #print(acc_dic)
    if step == 200:
        epsilon = 0.03
    if step == 1000:
        epsilon = 0.005
    if step == 10000: 
        epsilon = 0.0001
    #set all the other values to zero, could maybe use the pop command?
    acc_dic = [] 
    loss_dic = []
    elite_workers = []
   



    
    