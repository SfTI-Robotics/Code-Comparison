"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

"""
Policy gradient 

1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy πθ: S1,A1,R2,S2,A2,…,ST.
3. For t=1, 2, … , T:
    a. Estimate the the return Gt;
    b. Update policy parameters: θ←θ+αγtGt∇θlnπθ(At|St)

"""

import numpy as np
import tensorflow as tf

# randomisation 
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient: # this is the learning algorithm object
    def __init__( #constructor for object
            #variables for object to be initialised: need to be set
            self,
            n_actions,
            n_features,
            #variables for object to be initialised: fixed values
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        # set values for object variables from input of constructor
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # preallocate more object variables
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        #call function/method to run it
        self._build_net()

        #set variable as starting neural network, launcher for TF
        """ 
        A class for running TensorFlow operations.
        A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. 
        """
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following

            """
            Writes Summary protocol buffers to event files.

            The FileWriter class provides a mechanism to create an event file in a given directory and add summaries 
            and events to it. The class updates the file contents asynchronously. This allows a training program to call
            methods to add data to the file directly from the training loop, without slowing down training.
            """
            tf.summary.FileWriter("logs/", self.sess.graph)

        #launch session 
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # name_scope: A context manager for use when defining a Python operation
        # A graph maintains a stack of name scopes. 
        #           with name_scope(...): statement pushes a new name onto the stack for the lifetime of the context.
        with tf.name_scope('inputs'):
            #define layers 
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        
        # fc1 --> fully connected or feature column

        """
        Functional interface for the densely-connected layer.

        This layer implements the operation: outputs = activation(inputs * kernel + bias) where activation
         is the activation function passed as the activation argument (if not None), kernel is a weights 
         matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True).
        """
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        #softmax calculates the probabilities distribution of the event over ‘n’ different events. 
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            # minimise loss using adam optimiser (since it is more sophisticated than gradientDescentOptimiser)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #local variable created 
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  
        # select action w.r.t the actions pro, randomly -> stochastic
        return action

    # trajectory or experience
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={

            # add episodic obs, actions, rewards into layers 
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs], Stack arrays in sequence vertically (row wise).
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data 
        return discounted_ep_rs_norm

    # calculate return 
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
