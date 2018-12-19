
# DQN Code comparison 

Code used:

### Greg - Keras

[Article](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)|[Code](https://github.com/gsurma/cartpole)

### Morvan - Tensorflow


### CN Blogs - Keras

## Overall Differences

### Diff 1: Keras vs. Tensorflow
Keras is a package based upon Tensorflow. It allows for much easier implementation but also has less flexibility/customisation than Tensorflow.

### Diff 2

 

{ Format }

[constructor]
1. initialise instance variables
2. call build net function
3. get parameters
4. instantiate session
5. launch session

[build network]
\\ evaluation net
1. initialise placeholders for state and q-target
2. initialise weight and bias paramters
    \\ layer 1
    1. get weight and bias variable using get_variable()
    2. set up relu (recitified linear unit) neural network on current state

    \\ layer 2 
    1. get weight and bias variable using get_variable()
    2. calculate q-evaluation

\\ loss
 find mean of error squared

\\ train
 minimise loss using RMS Prop optimiser

\\ target network
1. set up collection for target
\\ layer 1
    1. get weight and bias variable using get_variable()
    2. set up relu (recitified linear unit) neural network on next state
\\ layer 2
    1. get weight and bias variable using get_variable()
    2. calculate next q-value 

[store transition]
1. if 'memory counter' does not exist
        create this variable
2. store transition <s, a, r, s'> in horizontal stack
3. replace old memory with new memory
4. increment counter

[choose action]
1. add new dimension to observations
2. if random num < e-greedy
        choose action greedily, exploit
    else 
        choose action randomly, explore

[learn]
1. replace target parameters every 300 iterations
2. randomly sample transition history through 
3. find cost 
4. increase epsilon

[plot graph]


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

#### CN Blogs - Keras
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

#### CN Blogs - Keras

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

##### CN Blogs - Keras

He initialises the memory under a different class from the brain

##### Morvan - Tensorflow

```
self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
```

The memory is initialised as an array of zeros with rows being the memory size (=500) and the number of features (=2) multiplied by 2 plus 2 (this is for the six elements in a transition <s_x. s_y, a, r, s'_x, s'_y>). 


#### Building model

##### Greg - Keras
Using Keras, the 
```
# create neural network by stacking the layers in a linear order
self.model = Sequential()
self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
self.model.add(Dense(24, activation="relu"))
self.model.add(Dense(self.action_space, activation="linear"))
# if weights are not specified, default is sample weight
self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
```

##### CN Blogs - Keras



##### Morvan - Tensorflow


### Memory/Store Transition 


### Experience Replay

### Choose Action




