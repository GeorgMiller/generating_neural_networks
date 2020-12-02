import tensorflow as tf
import gym
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense

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

class Agent(object):

    def __init__(self, name, env, network, input_shape, output_dim):
        """Agent worker thread

        Args:
            name (str): Name of this agent (usually, "thread-{id}")
            env (gym.Env): Gym environment
            network (A3CNetwork): Actor Critic Network
            input_shape (list): A list of [H, W, C]
            output_dim (int): Number of actions
        """
        self.name = name
        self.env = env
        self.network = network
        self.input_shape = input_shape
        self.output_dim = output_dim


    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        print(message)

    def play_episode(self):
        

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005, rho=0.99)


        #self.env.render()
        states = []
        actions = []
        rewards = []
        states_2 = []
        values_v = []
        action_pred = []

        s = self.env.reset()
        s = pipeline(s)
        current_state = s

        done = False
        total_reward = 0
        time_step = 0

        episode_length = 0
        while not done:

            self.env.render()

            with tf.GradientTape() as tape:

                with tf.GradientTape() as tape2:

                    states_for_network = np.reshape(current_state, [-1, *self.input_shape]) #what does the star before self mean?
            
                    output = self.network(states_for_network.astype('float32'))
                    v = output[1]
                    #print(np.squeeze(action[0]),"this is the action") #TODO: here i need to change it
                    

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
                        if time_step >= 5 or done:

                            #print(output)
                            
                            states = tf.convert_to_tensor(states) #np.array(states)
                            actions = tf.convert_to_tensor(actions,dtype='int32')
                            actions = tf.math.subtract(actions, tf.convert_to_tensor(1)) # np.array(actions) - 1
                            rewards = tf.convert_to_tensor(rewards) #np.array(rewards)
                            states_2 = tf.convert_to_tensor(states_2)
                            values_v = tf.reshape(values_v, [-1])
                            values_v = tf.convert_to_tensor(values_v)
                            action_pred = tf.reshape(action_pred, [-1,3])
                            action_pred = tf.convert_to_tensor(action_pred)

                        
                            #values_pred = np.squeeze(values_pred[1])
                            #print(values_pred) #TODO:this also needs to be changed
                            rewards = discount_reward(rewards, gamma=0.99)
                            rewards = tf.convert_to_tensor(rewards)
                        
                            
                            '''
                            advantage = tf.math.subtract(rewards, values_v)
                            advantage = advantage - tf.math.reduce_mean(advantage)
                            advantage = tf.math.divide(tf.math.add(tf.math.reduce_std(advantage), tf.convert_to_tensor(1e-7)), advantage)
                            '''
                            advantage = rewards - 0.99*values_v #- values_v[:-1]
                            #ones all theses values are calculated one can finally update the network
                            
                            #function needed for forward pass through network with the states

                            #output = self.network(states)
                            
                            action_onehot = tf.one_hot(actions, self.output_dim) #this makes the new shape
                            #print('this is the action onehot thing')
                            #self.network.output[0]
                            single_action_prob = tf.reduce_sum(action_pred * action_onehot, axis = 1) 

                            #print(single_action_prob,'this is the single action prob')
                            #now i have the single action probability of the network
                            entropy = action_pred * tf.math.log(action_pred + 1e-7) # I calculate the entropy of the action probabilities
                            entropy = - tf.reduce_sum(entropy,axis=1)

                            log_action_prob = tf.math.log(single_action_prob + 1e-7)
                            maximize_objective = log_action_prob * advantage
                            
                            #this is the loss for the actor part of the network
                            actor_loss = - tf.reduce_sum(maximize_objective)
                            
                            

                            # this is the value network
                            #values = output[1]    #self.network.output[1]
                            values_loss = tf.reduce_sum(tf.math.squared_difference(rewards, values_v))
                            


                            total_loss = tf.math.add(actor_loss, values_loss*0.5)
                            total_loss = tf.math.subtract(total_loss, entropy * 0.01)
                            #print('this is the total loss',total_loss)
                            # remove from calculation
                            
                            grads = tape.gradient(actor_loss, self.network.trainable_variables)
                            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
                            #print('these are the grads for tape1',grads)
                            
                            

                            grads = tape2.gradient(values_loss, self.network.trainable_variables)
                            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
                            #print('these are the grads for tape2',grads)
                    
                        

        

                            #reset all the states
                            states, actions, rewards, values_v, states_2, action_pred = [], [], [], [], [], []
                            time_step = 0



        self.print(total_reward)

        return total_reward


def training_thread(agent, global_network, env, thread,lock):

    #global global_network

    print('thread has started', thread)
    #lock.acquire()
    reward = agent.play_episode(lock)
    env.close()
    #lock.release()
    print('this is the reward and the thread', reward, thread)



def main():

    env = gym.make("Pong-v0")

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
    

    network = tf.keras.Model(inputs=states,outputs=[action_prob, values])
    global_network = network

    print(network.summary())
    
    agent = Agent("new_agent",env,network,input_shape=[80,80,1], output_dim=3)
    print(agent)
    
    for i in range(10000):

        i = i+1
        reward = agent.play_episode()
        agent.print(reward)
        print("episode:", i)
        
'''
        
lock = Lock()
        n_threads = 2#multiprocessing.cpu_count()
        print(n_threads)
        envs = [gym.make("Pong-v0") for i in range(n_threads)]
        networks = [network for i in range(n_threads)]
        agents = [Agent("new_agent",envs[i],networks[i],input_shape=[80,80,1], output_dim=3) for i in range(n_threads)]

        #p = multiprocessing.Process(target=training_thread,args=(agent,env,1))
        #p.start

        threads = [threading.Thread(target=training_thread,daemon=True,args=(agents[i], global_network, envs[i], i,lock)) for i in range(n_threads)]
        print(threads)
        for t in threads:
            time.sleep(2)
            t.start()
        
        for t in threads:
            time.sleep(10)
            t.join()





        with tf.device("/cpu:0"): 
            global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
            #trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = network #AC_Network(s_size,a_size,'global',None) # Generate global network
            num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
            workers = []
            # Create worker classes
            for i in range(num_workers):
                workers.append(Worker(DoomGame(),i,s_size,a_size,optimizer,model_path,global_episodes))
            saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
                
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate threat.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)





    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ['ps0.example.com:2222'], "worker": ['worker0.example.com:2222']})

    # Create and start a server for the local task.
    server = tf.distribute.Server(cluster,
                             job_name='job_name',
                             task_index='task_name')

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(FLAGS.task_index),
                cluster=cluster)):
            network = A3CNetwork("task_{}".format(FLAGS.task_index), input_shape=[80, 80, 1], output_dim=3)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000), ]

        env = gym.make("Pong-v0")

        if FLAGS.task_index == 0:
            env = gym.wrappers.Monitor(env, "monitor", force=True)

        agent = Agent("task_{}".format(FLAGS.task_index), env, network, [80, 80, 1], output_dim=3)
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="train_logs",
                                               save_summaries_secs=None,
                                               save_summaries_steps=None,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                reward = agent.play_episode(mon_sess)
                agent.print(reward)

    '''
main()
   