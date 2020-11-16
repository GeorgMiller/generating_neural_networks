import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense


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
    

batch_size = 32
epochs = 2
num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("dfghjkjhgvfcdghjkl", y_train.shape)
x_train = np.reshape(x_train, (-1, 28,28,1))
x_test = np.reshape(x_test, (-1, 28,28,1))

x_train = x_train / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

optimizer = keras.optimizers.Adam()

# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy()

mnist_model = MNIST_model()

#mnist_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#mnist_model.fit(train_dataset, batch_size=batch_size, epochs=epochs)


epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            logits = mnist_model(x_batch_train)
            

            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, mnist_model.trainable_weights)

        
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * 64))




    
