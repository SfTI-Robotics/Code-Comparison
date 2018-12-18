# Q-learning

### Frozen lake 
Arthur[Article](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) | [Code](https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb)

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

# STATE_BOUNDS[0] = [STATE_BOUNDS[0][0]/2, STATE_BOUNDS[0][1]/2]    # x
STATE_BOUNDS[1] = [-0.5, 0.5]                                       # x dot
# STATE_BOUNDS[2] = [STATE_BOUNDS[2][0]/2, STATE_BOUNDS[2][0]/2]    # theta
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]             # theta dot

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.2

# Defining the simulation related constants
NUM_EPISODES = 1000
MAX_T = 250             # time-steps
STREAK_TO_END = 120
SOLVED_T = 199

# ignore these 2 lines its for printing
DEBUG_MODE = False 
ENABLE_UPLOAD = False
```

- The states are split into buckets in order to make them discrete, because Q Learning algorithms cannot be used with continuous states???
- `env.action_space.n` returns the length of the action space ,how many actions you can take in this case it would be 2 either left or right.

- `list()`, `zip()`: see python syntax in website but they create a list of pairs for each state (x, xDOT, theta,  thetaDOT) showing the lower and upper bounds that can be reached before an episode ends.

- state bounds for x and theta are already set by the environment, so only need to manually set xDOT and thetaDOT.

- streak totals the number of times it reaches the terminal state

- solved t = 199 because it is the least amount of time-steps for the episode to end (ie. pole goes out of bounds)
- a gamma variable isn't set here but rather set later in a function. Also we don't have  a decay rate set up here as this person creates a function that uses logarithmic operation to decrease epsilon later in the code.

### Q Table

#### Taxi

```
Q_table = np.zeros((env.observation_space.n, env.action_space.n))
```
#### Frozen 

Is the same as taxi

#### Cartpole
```
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS, ))
```
The `+` sign acts as a comma, forming state, action pairs. 

### Getting states

`state = env.reset()` returns the initial state.
In Cartpole, the continuous state is converted to a discrete state.

### Choosing actions

#### Taxi
```
xx_tradeoff = random.uniform(0, 1)

        if xx_tradeoff > exploration_rate:
            # exploitation
            action = np.argmax(Q_table[state,:])
        else:
            # exploration
            # [OpenAI Gym] sample: picking one value from array of values
            action = env.action_space.sample()
```

xx_tradeoff is a number between 0 and 1 and we use that number implement e-greedy. The exploration rate decays as the number of episodes increases. At the start, the algorithm does more exploration, then it gradually does more exploits to try and find the optimal policy.  

#### Frozen Lake
```
#Choose an action by greedily (with noise) picking from Q table
# noise is a random initialiser for Q-table
a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
```
using noise is a searching method.
In order for the initial Q-values to be non-zero, a random number is added on. This random number will decay as the number of episodes increases, due to the scaling factor(1/i+1). 

The selection of the next action is based on a greedy policy in the later episodes.


#### Cartpole
```
action = select_action(state_0, explore_rate)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action
 ```
Same as Taxi.

### Update Q-values

```
state_, reward, done, info = env.step(action)

# update Q table using algorithm
Q_table[state,action] += learn_rate * (reward + discount * (np.max(Q_table[state_, :]) - Q_table[state, action]))
```
All three code scripts use the same Bellman equation and gym `step()` function.

### Finish episode

#### Taxi

```
if done:
    break
```

Stops the episode when terminal state is reached
#### Frozen 
```
if d == True:
    os.system('clear')
    env.render()
    
    print("Episode ended")
    time.sleep(1)
    
    break
```

This is essentially the same as frozen lake except time sleep is used to prevent memory leak.

#### Cartpole
```
if done:
    print("Episode %d finished after %f time steps" % (episode, t))

    if t >= SOLVED_T:
        num_streaks += 1
    else:
        num_streaks = 0
    break
```
Checks if the episode was completed within the number of steps (ie. streak) and increments the number of streaks.


### Cumulative Rewards
The final reward is a normalised sum of the total rewards from each episode: `sum(cumulative_return)/episode_num`.
