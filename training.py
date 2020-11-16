import keras
import tensorflow as tf 
import numpy as np
import hypernetwork, MNIST_network

#from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()


batch_size = 32

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, (-1, 28,28,1))
x_test = np.reshape(x_test, (-1, 28,28,1))

x_train = x_train / 255
y_train = keras.utils.to_categorical(y_train, num_classes)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


z = np.random.uniform(low = -1, high = 1, size = 300)
z = z[np.newaxis,:]


def train_mnist_model():

    return None

@tf.function
def train_hypernetwork(dataset):

    hypernetwork_model = hypernetwork.Hypernetwork()

    input_shape = (32,28,28,1)
    mnist_model = MNIST_network.MNIST_model()
    

    mnist_model.build(input_shape)
    for layer in mnist_model.layers: 
        print(layer)
        layer.built = True

    print(mnist_model.summary())

    #tf.keras.backend.set_floatx('float64')
    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape(watch_accessed_variables=False) as tape:

                tape.watch(hypernetwork_model.trainable_variables)

                index_w1 = 800
                index_b1 = 832

                index_w2 = index_b1 + 12800
                index_b2 = index_b1 + 12816

                index_w3 = index_b2 + 6272
                index_b3 = index_b2 + 6280

                index_w4 = index_b3 + 80
                index_b4 = index_b3 + 90


                weights_pred = hypernetwork_model(z)

                print(weights_pred.shape)

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
         
                mnist_model.conv2D_1.kernel = kernel_1
                mnist_model.conv2D_1.bias = b_1

                mnist_model.conv2D_2.kernel = kernel_2
                mnist_model.conv2D_2.bias = b_2

                mnist_model.dense_1.kernel = kernel_3
                mnist_model.dense_1.bias = b_3

                mnist_model.dense_2.kernel = kernel_4
                mnist_model.dense_2.bias = b_4

                
                
                #mnist_model.layers[0].kernel = kernel_1
                #print("really?", mnist_model.dense_1.get_weights)
                logits = mnist_model(x_batch_train)
                print(logits, mnist_model.dense_2.kernel, "dfghjklkjhgfdfghjk")


                loss_value = tf.Variable(loss_fn(y_batch_train, logits))
              

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, hypernetwork_model.variables)



            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, hypernetwork_model.trainable_variables))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * 64))


train_hypernetwork(train_dataset)
