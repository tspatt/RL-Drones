# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:19:50 2019

@author: Thomas S. Patterson
"""
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model:
    #Intialization function: must pass the 
    # num of states in enviorment
    # num of possible actions
    
    def __init__(self, num_states, num_actions, batch_size):
        
        #Drone number of states should be 
        # 1. X Postion
        # 2. Y Postion
        # 3. Z Postion
        # 4. X Velocity
        # 5. Y Velocity
        # 6. Z Velocity
        # 7. X Angle
        # 8. Y Angle
        # 9. Z Angle
        #10. X Angular Velocity
        #11. Y Angular Velocity
        #12. Z Angular Velocity
        #13. Prev X Postion
        #14. Prev Y Postion
        #15. Prev Z Postion
        #16. Prev X Velocity
        #17. Prev Y Velocity
        #18. Prev Z Velocity
        #19. Prev X Angle
        #20. Prev Y Angle
        #21. Prev Z Angle
        #22. Prev X Angular Velocity
        #23. Prev Y Angular Velocity
        #24. Prev Z Angular Velocity
        self._num_states = num_states
        
        
        #Actions:
        # 1. Roll
        # 2. Pitch 
        # 3. Yaw 
        # 4. Thrust
        self._num_actions = num_actions
        self._batch_size = batch_size
        
        # define the placeholders
        self._states = None
        self._actions = None
        
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        
        # now setup the model
        self._define_model()
        
    #define model function: Sets up the model structure and main operations
    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        #Convolution Layer
        conv1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        #fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        #fc3 = tf.layers.dense(fc2, 50, activation=tf.nn.relu)
        #fc4 = tf.layers.dense(fc4, 50, activation=tf.nn.relu)
        #3c5 = tf.layers.dense(fc5, 50, activation=tf.nn.relu)
        
        self._logits = tf.layers.dense(logits, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()
        
    #return the output of the network, by calling the _logits operation, with an input of a 
    #single state
    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self.num_states)})
    # predict_batch predicts a whole batch of outputs when given a whole bunch of input 
    # states – this is used to perform batch evaluation of Q(s,a) and Q(s′,a′) values 
    #for training
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})
    
    # train_batch which takes a batch training step of the network
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})
        
class Memory:
    # Max memory will control the number of tuples _samples list can hold. 
    # bigger is better, as it ensures better data with less errors
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
        
    #takes and indiv tuple and appends it the list.
    #if the list is now too large, the first data point is pop()'d off
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
    
    #returns a random seletion of samples
    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
        
class GameRunner:
    #
    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._min_x_store = []
        self._min_y_store = []
        self._min_z_store = []

    
    #
    def run(self):
        
        #open ai jargon, gazebo adjust
        state = self._env.reset()
        tot_reward = 0
        #values need to be adjusted be starting distance of vector b/w drone and goal
        min_x = 0
        min_y = 0
        min_z = 0
        while True:
            if self._render:
                self._env.render()
    
            action = self._choose_action(state)
            
            #
            next_state, reward, done, info = self._env.step(action)
            """
            if next_state[0] >= 0.1:
                reward += 10
            elif next_state[0] >= 0.25:
                reward += 20
            elif next_state[0] >= 0.5:
                reward += 100
            """
            #the goal is to minimize the x,y,z disances
            if next_state[0] < min_x:
                max_x = next_state[0]
            if next_state[1] < min_y:
                max_x = next_state[1]
            if next_state[2] < min_z:
                max_x = next_state[2]
                
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None
    
            self._memory.add_sample((state, action, reward, next_state))
            self._replay()
    
            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                                      * math.exp(-LAMBDA * self._steps)
    
            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward
    
            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break
    
        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))
        
        def _choose_action(self, state):
            if random.random() < self._eps:
                return random.randint(0, self._model.num_actions - 1)
            else:
                return np.argmax(self._model.predict_one(state, self._sess))
            
        def _replay(self):
            batch = self._memory.sample(self._model.batch_size)
            states = np.array([val[0] for val in batch])
            next_states = np.array([(np.zeros(self._model.num_states)
                                     if val[3] is None else val[3]) for val in batch])
            # predict Q(s,a) given the batch of states
            q_s_a = self._model.predict_batch(states, self._sess)
            # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
            q_s_a_d = self._model.predict_batch(next_states, self._sess)
            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states))
            y = np.zeros((len(batch), self._model.num_actions))
            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]
                # get the current q values for all actions in state
                current_q = q_s_a[i]
                # update the q value for action
                if next_state is None:
                    # in this case, the game completed after action, so there is no max Q(s',a')
                    # prediction possible
                    current_q[action] = reward
                else:
                    current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q
            self._model.train_batch(self._sess, x, y)
    
if __name__ == "__main__":
    
    #open ai enviorment created
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    
    #number of states extracted from the enviorment itself
    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    #network object and memeory object created
    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    #tensor flow object created
    with tf.Session() as sess:
        sess.run(model.var_init)
        #gamerunner class created
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA) 
        
        #number of episodes defined
        num_episodes = 300
        cnt = 0
        
        #while loop runs each episode. It prints every 10th result
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1
            
        #total reward is plotted
        plt.plot(gr.reward_store)
        plt.show()
        plt.close("all")
        
        #max on the x axis is shown
        plt.plot(gr.max_x_store)
        plt.show()
    
    