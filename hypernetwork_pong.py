import numpy as np 
import tensorflow as tf 
import gym
import keras
import hypernetwork
from keras.layers import Conv2D, MaxPool2D, Dense

#Hyperparameters for the game of pong

input_shape = [80,80,1]
output_dim = 3


import multiprocessing

import threading
from threading import Thread, Lock

from PIL import Image  

import time

#from scipy.misc import imresize #check out this import and probably install it


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image

    (1) Crop image (top and bottom)
    (2) Resize to smaller image
    (3) Remove background & grayscale

    Args:
        image (3-D array): Numpy array of shape (H, W, C)
        new_HW (tuple): Target image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color as a tuple of (R, G, B)

    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)


    return image

def resize_image(image, new_HW):
    """Returns a resized image

    Args:
        image (3-D array): Numpy array of shape (H, W, C)
        new_HW (tuple): Target size (height, width)

    Returns:
        image (3-D array): Resized image (height, width, C)
    """

    image = Image.fromarray(image)
    image = image.resize(new_HW)
    return  np.asarray(image)
    #imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0

    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)

    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image

def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.

        However, when the reward is either -1 or 1,
        it means the game has been reset.

        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


env = gym.make('Pong-v0')

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

hypernet = hypernetwork.Hypernetwork(input_shape,output_shape)

states = tf.keras.Input((80,80,1),dtype='float32')
x = Conv2D(16, (8, 8), padding="same", activation="relu", dtype='float32')(states)
x = MaxPooling2D(pool_size=(2, 2), strides=(4,4), padding="same",dtype='float32')(x)
x = Conv2D(32, (4, 4), padding="same", activation="relu", dtype='float32')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same",dtype='float32')(x)
x = tf.keras.layers.Flatten()(x)

action_prob = Dense(254, activation= "relu",dtype='float32')(x)
action_prob = Dense(3, activation= "softmax",dtype='float32')(action_prob)

values = Dense(254, activation= "relu",dtype='float32')(x)
values = Dense(1,dtype='float32')(values)


actor = tf.keras.Model(inputs = states, outputs = action_prob)
    
critic = tf.keras.Model(inputs = states, outputs = values)

def set_weights(network, weights):

    layer_1_shape = 1040
    layer_3_shape = 8224
    layer_6_shape = 813054
    layer_7a_shape = 765
    layer_7b_shape = 255
    
    network.layers[1].kernel = np.reshape(weights[:layer_1_shape], [80,80,16])
    network.layers[3].kernel = np.reshape(weights[layer_1_shape:layer_3_shape], [20,20,32])
    network.layers[6].kernel = np.reshape(weights[layer_3_shape:layer_6_shape], [-1,254])
    network.layers[7].kernel = np.reshape(weights[layer_6_shape:layer_7_shape], [-1,765])

    return network

def train_network(network):

    states = []
    actions = []
    actions_prob = []
    rewards = []
    values = []

    s = env.reset()
    env.render()

    s = pipline(s)
    current_state = s


    done = False
    total_reward = 0
    time_step = 0



    while not done:

        with tf.GradientTape() as tape:

            network_input = tf.reshape(current_state, [-1, input_dim])

            weights = hypernet(network_input)
            network = set_weights(weights)
            output = self.network(states_for_network.astype('float32'))
            v = output[1]
            a = np.random.choice(np.arange(self.output_dim) + 1, p=np.squeeze(output[0]))

            s2, r, done, _ = self.env.step(a)

            s2 = pipeline(s2)
            total_reward += r

            states.append(current_state)
            action_pred.append(output[0])
            actions.append(a)
            rewards.append(r)
            states_2.append(s2)
            values_v.append(v)

            current_state = s2 - s
            s = s2

            episode_length += 1

            if r == -1 or r == 1 or done:
                time_step += 1

            # here comes the training
                if time_step >= 10 or done:

                    states = tf.convert_to_tensor(states) #np.array(states)
                    actions = tf.convert_to_tensor(actions,dtype='int32')
                    actions = tf.math.subtract(actions, tf.convert_to_tensor(1)) # np.array(actions) - 1
                    rewards = tf.convert_to_tensor(rewards) #np.array(rewards)
                    states_2 = tf.convert_to_tensor(states_2)
                    values_v = tf.reshape(values_v, [-1])
                    values_v = tf.convert_to_tensor(values_v)
                    action_pred = tf.reshape(action_pred, [-1,3])
                    action_pred = tf.convert_to_tensor(action_pred)
                    
                    rewards = discount_reward(rewards, gamma=0.99)
                    rewards = tf.convert_to_tensor(rewards)
                    
                    advantage = tf.math.subtract(rewards, values_v)
                    advantage = advantage - tf.math.reduce_mean(advantage)
                    advantage = tf.math.divide(tf.math.add(tf.math.reduce_std(advantage), tf.convert_to_tensor(1e-7)), advantage)                 
                    action_onehot = tf.one_hot(actions, output_dim) 
                    single_action_prob = tf.reduce_sum(action_pred * action_onehot, axis = 1) 

                    entropy = (- action_pred) * tf.math.log(action_pred + 1e-7) # I calculate the entropy of the action probabilities
                    entropy = tf.reduce_sum(entropy,axis=1)

                    log_action_prob = tf.math.log(single_action_prob + 1e-7)
                    maximize_objective = log_action_prob * advantage + entropy * 0.005
                    actor_loss = - tf.reduce_sum(maximize_objective)
                    values_loss = tf.reduce_sum(tf.math.squared_difference(rewards, values_v))
                    total_loss = tf.math.add(actor_loss, values_loss*0.5)
                    
                    grads = tape.gradient(total_loss, hypernet.trainable_variables)
                    optimizer.apply_gradients(zip(grads, hypernet.trainable_variables))

                    states, actions, rewards, values_v, states_2, action_pred = [], [], [], [], [], []
                    time_step = 0
                    

    return total_loss
    


def update_hypernetwork(loss):
    pass