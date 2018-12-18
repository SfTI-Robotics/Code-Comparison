
# DQN Code comparison 

Code used:

### Greg

[Article](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)|[Code](https://github.com/gsurma/cartpole)

### Morvan


### CN blogs

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

#### Greg 
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

#### CN Blogs
Same as Greg.

#### Morvan
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

### __init__ /Object constructor
```

```
#### Memory setup


#### building model


### Memory/Store Transition 


### Experience Replay

### Choose Action

### 


