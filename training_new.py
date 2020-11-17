
# Import tensorflow and check version
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import hypernetwork
import MNIST_network
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense


print('tensorflow version: {}'.format(tf.__version__))
tf.keras.backend.clear_session()



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# convert to float32 and normalize. 
x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32')   /255

# one-hot encode the labels 
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# add a channel dimension to the images
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

'''
# Define image dimensions
img_h = 28
img_w = 28
img_c = 1


infer_model = tf.keras.models.Sequential(name='infer_model')
infer_model.add(tf.keras.layers.Input(shape=(img_h, img_w, img_c), name='input_x' ))
infer_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu') )
infer_model.add(tf.keras.layers.MaxPool2D() )
infer_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu') )
infer_model.add(tf.keras.layers.MaxPool2D() ) 
infer_model.add(tf.keras.layers.Flatten() )

infer_model.add(tf.keras.layers.Dense(10, activation= 'softmax', name='out_layer') )

infer_model.summary()

'''

infer_model = tf.keras.models.Sequential([
    Conv2D(32, (5, 5), padding="same", activation="relu", input_shape = (28,28,1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
    Conv2D(16, (5, 5), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
    tf.keras.layers.Flatten(),
    Dense(8, activation= "relu"),
    Dense(10, activation= "softmax")
])

def set_parameters(mnist_model, weights_pred):


    index_w1 = 800
    index_b1 = 832

    index_w2 = index_b1 + 12800
    index_b2 = index_b1 + 12816

    index_w3 = index_b2 + 6272
    index_b3 = index_b2 + 6280

    index_w4 = index_b3 + 80
    index_b4 = index_b3 + 90

    #These are the weights for the first layer: input is 15*32=480 and output is 26*32=832
    #So for the kernel we use 800 which has to be reshaped to (5,5,1,32) of the weights and for the bias 32
    

    kernel_1 = weights_pred[:index_w1]
    kernel_1 = tf.reshape(kernel_1,(5,5,1,32))
    b_1 = weights_pred[index_w1:index_b1]
    #b_1 = tf.reshape(b_1, (-1))
    
    #These are the weights for the second layer: input here is 15*16=240 and the outout is 16*801=12816
    #The kernel has to be reshaped to (5,5,1,32) and 16 for the bias

    kernel_2 = weights_pred[index_b1:index_w2]
    kernel_2 = tf.reshape(kernel_2,(5,5,32,16))
    b_2 = weights_pred[index_w2:index_b2]
    #b_2 = tf.reshape(b_2, -1)

    #These are the weights for the third layer: takes as input 15*8=120 and the output is 8*785=6280
    #The bias is 8 and the kernel shape is

    kernel_3 = weights_pred[index_b2:index_w3]
    kernel_3 = tf.reshape(kernel_3,(784,8))
    b_3 = weights_pred[index_w3:index_b3]
    #b_3 = tf.reshape(b_3, -1)
    
    #And these are the weights for the output layer

    kernel_4 = weights_pred[index_b3:index_w4]
    kernel_4 = tf.reshape(kernel_4,(8,10))
    b_4 = weights_pred[index_w4:index_b4]
    #b_4 = tf.reshape(b_4, -1)

    #print(mnist_model.conv2D_1.bias)

    mnist_model.layers[0].kernel = kernel_1
    mnist_model.layers[0].bias = b_1

    mnist_model.layers[2].kernel = kernel_2
    mnist_model.layers[2].bias = b_2

    mnist_model.layers[5].kernel = kernel_3
    mnist_model.layers[5].bias = b_3

    mnist_model.layers[6].kernel = kernel_4
    mnist_model.layers[6].bias = b_4

       


#infer_model = MNIST_network.MNIST_model()
#infer_model.built = True
hyper_model_x = hypernetwork.Hypernetwork()

def parameterize_model(model, weights):
    # function to parametrizes all the trainable variables of model using the stream of weight values in weights
    # This assumes weights are passed a single batch.
    weights = tf.reshape( weights, [-1] ) # reshape the parameters to a vector
    
    last_used = 0
    for i in range(len(model.layers)):

        # check to make sure only conv and fully connected layers are assigned weights.
        if 'conv' in model.layers[i].name or 'out' in model.layers[i].name or 'dense' in model.layers[i].name: 
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            model.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if model.layers[i].use_bias:
              weights_shape = model.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              model.layers[i].bias = new_weights
              last_used += no_of_weights

# Define accuracy metrics for validation
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-3) 




loss_accum = 0.0
batch_size = 32
for step in range(1, 6001):
  idx = np.random.randint(low=0, high=x_train.shape[0], size=batch_size)
  x, y = x_train[idx], y_train[idx]
  #print(y)
  
  with tf.GradientTape() as tape:
    # Predict weights for the infer model
    #print(x.shape)
    z = np.random.uniform(low = -1, high = 1, size = 300)
    z = z[np.newaxis,:]
    #print(z[:,0:10])

    generated_parameters = hyper_model_x(z)
    #generated_parameters = generated_parameters[np.newaxis,:]

    set_parameters(infer_model, generated_parameters)    
    
    # Inference on the infer model
    preds = infer_model(x)
    #print("these are the predictions", generated_parameters.shape)
    loss = loss_fn( y, preds)
    #print(loss)
    loss_accum += loss
    train_acc_metric( y, tf.expand_dims(preds, 0)) # update the acc metric
    '''
    if step % 100 == 0: 
      
      var = generated_parameters.numpy()
      print('statistics of the generated parameters: '+'Mean, {:2.3f}, var {:2.3f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))
      for val_step in range(500): # 
        idx = np.random.randint(low=0, high=x_test.shape[0], size=batch_size)
        x, y = x_test[idx], y_test[idx]
        generated_parameters = hyper_model_x(z)
        generated_parameters = generated_parameters[np.newaxis,:]
        parameterize_model(infer_model, generated_parameters)    
        preds = infer_model(x)
        train_acc_metric( y, tf.expand_dims(preds, 0)) # update the acc metric
      print('\n Step: {}, validation set accuracy: {:2.2f}     loss: {:2.2f}'.format(step, float(train_acc_metric.result()), loss_accum))
      loss_accum = 0.0
      #train_acc_metric.reset_states()
    '''     
    if step % 100 == 0:
      var = generated_parameters.numpy()
      print('statistics of the generated parameters: '+'Mean, {:2.3f}, var {:2.3f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))
      print('\n Step: {}, validation set accuracy: {:2.2f}     loss: {:2.2f}'.format(step, float(train_acc_metric.result()), loss_accum))
      loss_accum = 0.0

    # Train only hyper model
    grads = tape.gradient(loss, hyper_model_x.trainable_weights)
    #print(grads)
    optimizer.apply_gradients(zip(grads, hyper_model_x.trainable_weights))


