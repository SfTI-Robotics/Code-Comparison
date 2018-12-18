# Q-learning

### Frozen lake 
Arthur Juliani [Article](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) | [Code](https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb)

### Cart-pole
Matt [Article](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947) | [Code](https://github.com/MattChanTK/ai-gym/blob/master/cart_pole/cart_pole_q_learning_theta_only.py)

### Taxi 
Thomas [Article](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe) | [Code](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb)

## Overall Differences

### Diff 1 : Abstraction

Frozen Lake and Taxi do not use any encapsulation/abstraction due to the simple nature of their environments. Cartpole requires additional functions to sort out the state into discrete categories, so with abstraction the code would get pretty long and unreadable.

### Diff 2: Training (learning) and testing models

Taxi had two sections one for training the model and one for testing. The training phase is where the agent is learning and creating a model with the optimal Q-values. The agent runs through the algorithm for 50000 episodes and selects actions based on an e-greedy policy. In the testing phase, the agent completes a further 100 episodes, however, it does not learn anymore (ie. doesn't update Q-values) and the actions are selected based on a greedy policy (choosing q values and hence best path for the Taxi to traverse).

Whereas in FrozenLake and Cartpole they only do learning. The training section isn't needed, it's only used to see the result of learning.

## In-depth comparison

### General Code Format
   
1. import dependencies  
2. initialise hyperparameters
3. initalisation of Q table
4. learning algorithm for training
    - for every episode
        - get start state
        - for every step
            - show rendered env (optional)
            - choose action
            - update Q table, rewards, state
5. use trained model to play game (optional)

### Parameters

1. Number of episodes is set to a very large number so there is sufficient amount of training and exploration
2. Number of steps is commonly 99, this is the maximum amount of steps the agent can take before the episode ends, regardless of if the terminal state is reached. 
3. Learning rate (alpha) is used in the Bellman equation
4. Discount is the gamma variable used in the Bellman equation
5. Exploration rate (1 - epsilon) sets the e-greedy probability of choosing a random action
6. Decay rate is used to slowly decrease the exploration rate so it exploits more towards the end 
   
#### Taxi
```
# hyperparameters 
episode_max = 50000
steps_max = 99 #num of steps max in each episode

learn_rate = 0.5 # alpha
discount = 0.5   # gamma

# exploration parameters
exploration_rate = 1.0  # epsilon
decay_rate = 0.01 #used to decrease epsilon
```


#### Frozen Lake
```
# Set learning parameters
lr = .8 # alpha
y = .95 # gamma
num_episodes = 2000
decay_rate = 0.01 #used to decrease epsilon
```

A maximum steps variable isn't defined but rather hardcoded into the while loop. Its also doesn't have an exploration rate as doesn't use e-greedy.

#### Cart Pole
```
## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta') 
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Manually setting bounds (needed for the x_dot and theta_dot)
#STATE_BOUNDS[0] = [STATE_BOUNDS[0][0]/2, STATE_BOUNDS[0][1]/2]
STATE_BOUNDS[1] = [-0.5, 0.5]
#STATE_BOUNDS[2] = [STATE_BOUNDS[2][0]/2, STATE_BOUNDS[2][0]/2]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.2

# Defining the simulation related constants
NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = False
ENABLE_UPLOAD = False
```

The states are split into buckets in order to make them discrete, because Q Learning algorithms cannot be used with continuous states???
`env.action_space.n` returns the length of the action space ,how many actions you can take in this case it would be 2 either left or right.

list zip 
