# DQN 

"""
QUESTIONS?????

1. why actions = shape[None, ]
2. replacing q-target with a, r, s' placeholders in modified file
3. syntax for activation = relu
4. stop_gradient: backpropagate without q-target
5. cost vs. loss



"""

import gym
import numpy as np
import tensorflow as tf

# create object class for learning algorithm
class DQN:
    def _init_(self,n_feature,n_action):
        # assign variables
        self.lr=0.01
        self.epsilon=0.9
        self.gamma=0.9
        self.memory_size=500
        self.batch_size=32
        self.n_action=n_action
        self.n_feature=n_feature
        self.learning_counter=0
        self.memory=np.zeros((self.memory_size,n_feature * 2 + 2))
        
        self.sess=tf.Session()

        # launch tensorflow session
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # set up variables 
        self.state = tf.placeholder(tf.float32, shape = [None, self.n_feature], name = 'state')
        self.action = tf.placeholder(tf.int32, shape = [None, ], name = 'action')
        self.reward = tf.placeholder(tf.float32, shape = [None, ], name = 'reward')
        self.nextState = tf.placeholder(tf.float32, shape = [None, self.n_feature], name = 'nextState')
        
        # weight: generate random number from normally distributed samples 
        weight = tf.random_normal_initializer(0.,0.3)
        # bias: generate constant value
        bias = tf.constant_initializer(0.1)

        # evaluation network: create layers of 
        with tf.variable_scope('eval_net'):
            eval1 = tf.layers.dense(self.state, 20, activation = tf.nn.relu, kernel_initializer = weight, bias_initializer = bias, name='eval1')
            self.q_eval = tf.layers.dense(eval1, self.n_actions, activation = 'relu',kernel_initializer = weight, bias_initializer = bias, name='q_eval')

        # target network
        with tf.variable_scope('target_net'):
            target1 = tf.layers.dense(self.nextState, 20, activation= 'relu',kernel_initializer = weight, bias_initializer = bias, name= 'target')
            self.q_next = tf.layers.dense(target1, self.n_actions, activation= 'relu',kernel_initializer = weight, bias_initializer = bias, name=)

        # calculate q-target
        with tf.variable_scope('q_target'):
            q_target=self.r+self.gamma*tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target=tf.stop_gradient(q_target)

        # calculate q-evaluate: 
        with tf.variable_scope('q_eval'):

            # tf.stack(): combine tensors into one tensor, 
            #               increases dimension of tensor by concatenating arrays side-by-side
            action_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)

            # tf.gather_nd: able to access more dimensions of the tensor
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=action_indices)    # shape=(None, )
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval))


# import environment
env = gym.make('MountainCar-v0')

# variables
actions = env.action_space.n
states = env.observation_space.shape[0]

RL = DQN(states, actions)

for episode in range(1000):
    state = env.reset()

    for step in range(100):


        action = RL.chooseAction(state)

        stateNext, reward, isEpisodeFinished, info = env.step(action)

        RL.storeTransition(state, action, reward, stateNext)

        # update Q-values
        if step > 75:
            if step % 5 == 0:
                # experience replay
                RL.learn()

        if isEpisodeFinished:
            break

        state = stateNext

env.close()        



