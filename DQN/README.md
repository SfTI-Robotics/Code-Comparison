
# DQN Code comparison 

Code used:

### Keras

Greg  [Article](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288) | [Code](https://github.com/gsurma/cartpole)

Jaromír [Article](https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/) | [Code](https://github.com/jaromiru/AI-blog/blob/master/CartPole-basic.py)
### Tensorflow

Morvan  | [Code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5_Deep_Q_Network)



## Overall Differences

### Diff 1: Keras vs. Tensorflow
Keras is a package based upon Tensorflow. It allows for much easier implementation but also has less flexibility/customisation than Tensorflow. A layout of how Tensorflow is used to program a DQN: 

**[constructor]**
1. initialise instance variables
2. call build net function
3. get parameters
4. instantiate session
5. launch session

**[build network]**

**evaluation net**
1. initialise placeholders for state and q-target
2. initialise weight and bias paramters
    **layer 1**
    1. get weight and bias variable using get_variable()
    2. set up relu (recitified linear unit) neural network on current state

     **layer 2** 
    3. get weight and bias variable using get_variable()
    4. calculate q-evaluation

**loss**

 find mean of error squared

 **train**

 minimise loss using RMS Prop optimiser

**target network**
1. set up collection for target
   
    **layer 1**

    1. get weight and bias variable using get_variable()
    2. set up relu (recitified linear unit) neural network on next state
    
    **layer 2**

    1. get weight and bias variable using get_variable()
    2. calculate next q-value 

**[store transition]**
1. if 'memory counter' does not exist
        create this variable
2. store transition <s, a, r, s'> in horizontal stack
3. replace old memory with new memory
4. increment counter

**[choose action]**
1. add new dimension to observations
2. if random num < e-greedy
        choose action greedily, exploit
    else 
        choose action randomly, explore

**[learn]**
1. replace target parameters every 300 iterations
2. randomly sample transition history through 
3. find cost 
4. increase epsilon

**[plot graph]**

## In Depth Comparison

### Hyperparameters

#### Greg - Keras 
```
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
```
Memory size is used to set the maximum length of the memory array which stores the batch sample of transitions.

Batch size is used for experience replay. It determines the sample size of the stored transitions used in the learning algorithm. 

#### Jaromír - Keras
Same as Greg - Keras.

#### Morvan - Tensorflow
```
# input to  __init__():
learning_rate=0.01,
reward_decay=0.9,
e_greedy=0.9,
replace_target_iter=300,
memory_size=500,
batch_size=32,
e_greedy_increment=None

# instance variables
self.lr = learning_rate
self.gamma = reward_decay
self.epsilon_max = e_greedy
self.replace_target_iter = replace_target_iter
self.memory_size = memory_size
self.batch_size = batch_size
self.epsilon_increment = e_greedy_increment
self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
```
Setting up the hyperparameters in this way is slightly excessive since the variables are constant for every inistantiation of the object. Using this format, the hyperparameters must be assigned to an instance variable (ie. `self.__`) so that it is stored as a variable of the object. 

replace_target_iter is used so that after every x iterations the target parameters are updated

## DQNsolver/ Brain

### __init__ /Object constructor

#### Greg - Keras

Here he assigns action space to self and also intialises the expolration rate at its max value. Then he initalises his memory and neural network using keras' model and layers package.

#### Jaromír - Keras

The init functions assigns the state and action as a variable of the class/object. The neural network model is also initialised by calling the class function `_createModel()`.
#### Morvan - Tensorflow
As above, the states and actions are assigned to the object. Since the code uses Tensorflow, the variables for hyperparameters are set as constant inputs to the constructor to allow for flexibility/customisation. Then they intialise the memory which is detailed below. Next he goes on to build then network by calling the class function `_build_net()` how the network is built is detailed below.

Then 2 collection variables, t_params and e_params, are initialised. They are later used in the update of the target and evaluation networks. 
```
t_params = tf.get_collection('target_net_params')
e_params = tf.get_collection('eval_net_params')
```

After every x iteration the target parameters are replace by the evaluation parameter and he puts this in intialisation so that it can be called whenever needed as we will see later on. However there's no partcular reason its placed in initialisation it can be put anywhere.
```
self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
```
The computational graph is assigned to `sess` through the Session() function. 

An `if` statement for the summary protocol allows a training program to call methods without interfering with the training process (ie. does not slow it down).

The graph is then launched by the `run()` function which computes the initial values for all existing global variables. 

The cost history list is initialised as an empty list. 

#### Memory setup

##### Greg - Keras

```
self.memory = deque(maxlen=MEMORY_SIZE)
```
The memory is initialised as a deque data structure with a fixed length.

##### Jaromír - Keras

He initialises the memory under a different class from the brain

##### Morvan - Tensorflow

```
self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
```

The memory is initialised as an array of zeros with rows being the memory size (=500) and the number of features (=2) multiplied by 2 plus 2 (this is for the six elements in a transition <s_x. s_y, a, r, s'_x, s'_y>). 


#### Building model

##### Greg - Keras

```
# create neural network by stacking the layers in a linear order
self.model = Sequential()
self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
self.model.add(Dense(24, activation="relu"))
self.model.add(Dense(self.action_space, activation="linear"))
# if weights are not specified, default is sample weight
self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
```
Using Keras, the model is initialised as a sequence of densely-connected layers.To see what exactly the dense functio does see documentation.
This gives us a neural network of 4 layers. The first layer consists of 4 input nodes(the 4 observations) then 2 hidden layers both with 24 nodes lasty the output layer with two nodes that outputs which action to take left or right. Last line simply compiles the all the layers into one model that uses the [adam optimiser](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) to estimate values and calulates loss by mean square error this is when you square the loss and then take the average.

##### Jaromír - Keras

```
def _createModel(self):
    model = Sequential()
    model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
    model.add(Dense(output_dim=actionCnt, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)
    return model
```
The model is created as a function within the DQN algorithm class and is called in the class constructor.This is extremely similar to gregs example and this is because keras is a simple package with not alot of room for variation and customisation. But the number of layers here can be changed with only 3 layers input(4 nodes) and a hidden layer with 64 nodes and an output layer. They also uses a different optimiser [RMSProp](https://www.coursera.org/lecture/deep-neural-network/rmsprop-BhJlm)

##### Morvan - Tensorflow
Using Tensorflow, 2 networks are created, the evaluation and target. In the evaluation network, a graph is initialised for the state and Q-target, which hold the parameters used for its computation. The state graph has the shape = `[None, self.n_features]` and Q-target graph has the shape = `[None, self.n_actions]`.

In the scope `'eval_net'`, the weight and bias is initialised randomly. The first layer in the network w1 and b1 are created using the `get_variable()` function. The matrix multiplication of the state and w1 
A rectified linear unit (relu) layer is created with the matrix multiplication of the state and w1 

### Memory/Store Transition 

#### Greg
```
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
```
The function appends the transition to the memory deque. Once the capacity is reached, the deque automatically removes the oldest transition. 

#### Jaromir
```
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
```
He creates an entire class and he has several functions within it. He creates the samples array to store all the transitions, then uses the constructor to set the capacity of the array.
Then adds new transitions by appending when calling the add function, if its at its max capacity then it adds and then takes off the oldest memory. 

#### Morvan
```
def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

```
Using a counter, the array is updated with new transitions to meet the capacity constraints. 

### Experience Replay

#### Greg
```
   def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                # bellman equation
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # fit(x, y): trains the model for a given number of iterations (epochs) on a data set
            # verbose: displays info
            self.model.fit(state, q_values, verbose=0)
        # update exploration rate
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN,     self.exploration_rate)
```
He takes a random batch of 20 the default value of q update becomes just the current reward. Then checks the agent if it is in the terminal state currently. If it is not then the bellman is applied and updates the current q value. Else if it is int the terminal then  q_update remains in as the current reward which is always 1 in the terminal state. The q_update is stored with the corresponding action in the q_value matrix. Then the graph is trained using the q_values and current state through the `fit()` function. The exploration rate is decayed and checked if it is above the minimum rate. 

As you can see the q_update is updated twice this could be replaced with an if else statement however this format uses les computational power than if else and is actually good especially for larger environments.


#### Jaromir
```
def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)     # Q-value
        p_ = agent.brain.predict(states_)   # Q-next

        x = numpy.zeros((batchLen, self.stateCnt))      #state 
        y = numpy.zeros((batchLen, self.actionCnt))     #Q-value
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]    # Q-update
            if s_ is None:      # reached terminal
                t[a] = r
            else:
                # Bellman 
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s    
            y[i] = t

        self.brain.train(x, y)      # train() uses model.fit()
```
A random sample of the stored transitions (batch = <s, a, r, s'>) is taken. For the first element in the batch array (ie. state), the values are stored in the state array. Similarly, for the next states, they are extracted through a for loop. Then these are inputted into the graph to get q values.  Then you create an x array for your states and y for q values that will be used to train the model at the end of experience replay.First it uses a for loop to assign the s,a,r,s' values in each batch(needs to be done if not using deque) The q_update is updated using the bellman equation if the terminal state is not reached, otherwise it is assigned the reward. The state and Q-update is used as inputs to train the model. 

As you can see this is not the best coding practice as the variable naming is terrible please avoid this. The memory is an object which could have been used to extract the transition elements <s, a, r, s'> so that the code is neater. 

He still manages to stake the exact same steps as Greg but this is a good example of 'bad'coding
#### Morvan

```
def learn(self):

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory, experience replay
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

       
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)#array of indexes 
        # astype(): Cast a pandas object to a specified dtype.
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        
        reward = batch_memory[:, self.n_features + 1]

        #  Q-target update
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
```
The target parameters are checked if an update in required. 
The random sample of transitions are taken with the size depending on the counter. Compute the evaluation and next Q-values by inputting the the memory batch (here the transition elements are extracted in the dictionary). The target Q-value is updated with respect to the eval Q-value. The indices of the batch is generated by creating an array with values from 0 to the batch size, increments of 1. The actions indices are generated by extracting the actions from the batch memory and using the astype() to get the indices as integers. The reward is also extracted from the batch memory. Q-target updated using the Bellman equation. The cost is generated by computing the graph using the loss equation defined in the build net function. The inputs are the state and q-target.


### Choose Action

#### Greg
```
    def act(self, state):
        # rand: create array of given shape and populate with random samples from a uniform distribution over (0, 1)
        if np.random.rand() < self.exploration_rate: # exploration
            #  randrange: generate numbers from a range, allows intervals between numbers
            return random.randrange(self.action_space)
        #  predict: computation performed in batches, updates q-values
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) # exploitation
```

This uses the same exploration vs exploitation tradeoff seen previously in q learning except this time the chosen action and corresponding state is used for training and the biggest q value is returned.In orde to use predict you must reshape(see tensorflow syntax page) the state which is done in greg's code but later on `state = np.reshape(state, [1, observation_space])`
#### Jaromir
```
def act(self, s):
    if random.random() < self.epsilon:
        return random.randint(0, self.actionCnt-1)
    else:
        return numpy.argmax(self.brain.predictOne(s))
```
The exploitation vs exploration trade-off (e-greedy) method is used to choose the next action. The greedy choice returns the action based on the maximum Q-value by reshaping the state and inputting into the `predict` function on the model. 

#### Morvan
```
def choose_action(self, observation):
    # to have batch dimension when feed into tf placeholder
    # newaxis: used to increase the dimension of the existing array by one more dimension, when used once. 
    observation = observation[np.newaxis, :]

    if np.random.uniform() < self.epsilon:
        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, self.n_actions)
    return action
```
A new dimension to added to the observation to reshape into a matrix. Then the action is chosen e-greedily. 