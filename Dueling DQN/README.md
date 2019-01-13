# Duelling DQN Code Comaprison

Cart Pole | [Code]()

Doom [Article](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)
 | [Code](https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb)


## Overall differences

The structure of the two codes varies from each as the abstraction is set up according to the author's individual styles. Cart Pole code has a lot more abstraction and encapsulation, so expect to see more objects/classes.
Another key difference is that the environments they use widely differ as DOOM uses images that need to be processed and will run through the neural network layers using convulutional layers. Whereas the pendulum doesnt require this and so the code will be shorter
## In depth comparison

### Create Environment

creating object and environment

#### Doom
Doom is quite a different environment that requires additional steps after calling the environment also you have to initialise the environment at the begiining of the code. 

Additionally you have to take in account the external buttons:

```
possible_actions = np.identity(7,dtype=int).tolist()
return game, possible_actions
```
Game has 7 buttons may some may  be pause button and other things but we only need 5. The action space is initialised as an 7 x 7 identity matrix.

e.g:possible_actions = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]...]

[1, 0, 0, 0, 0] may correspond to up and [0, 1, 0, 0, 0] may correspond to down

### Hyperparameters
#### Doom

```
state_size = [100,120,4] 
```
We have 4 channels because of the 4 frames stacked ont top of each other and 100x120 is just the image size.

```
max_tau = 10000 
```

The target network is updated only when the step numbers reach a certain number(10000). This is the concept of fixed Q-values.


### Nueral Network 

#### Frames (DOOM only)

##### Preprocessed

The pixels of a single frame are normalised so that each pixel has a similar distribution. this helps the stochastic gradient descent converges faster.

##### Stack frame

Only every 4 frames is considered because having only 1 frame doesn't give the neural network a sense of motion for the objects in the game.
We use deque to stack the frames every episode.First appending the frame on the deque and then stack the array of frames onto each other(4 dimensions). When full, the deque automatically removes the oldest one(smalest q values and least reliable).


```
stacked_state = np.stack(stacked_frames, axis=2)
```

We consider 2 stacks(axis=2) one with the initial state and one with the next state(each with 4 frames).

`stacked_state` is a stack data type and `stacked_frames` is a deque data type. 



### DDDQN Algorithm

#### Init
##### DOOM
Has 4 placeholders that will be used later on: inputs, importance sampling weights, actions and target Q-value.

##### Cart Pole

`predictions` is the predicted Q-value 

#### Building Model
##### DOOM

###### Convolutional NN
So it uses 3 convolutional layers. Each one uses the `conv2d` function to split the frame into parts eg:player, victims etc... 
Then we use elu network to restrict the data to a range and adds the layers.

###### NN
Here we build two separate NN, one for the value function and the second advantage function. The inputs for both are the flattened processed layers discussed above.Then the second layers for each take the previous layer and give the output .V(s) is the reward number and advantage is the action to take. This is where the dueling DQN is implemented.

##### Cart Pole

First we create the hidden layers where all the operations will be done using weights and biases

value layers only have num of states to 1 as each state can only have one value but advantage is always num of states to num of actions as there's different q values

Value and Advantage functions both use the hidden layer  by multiplying it with their respective weights and adding on their respective biases.

In Output layer we use the formula value+(advantage-error) to solve backpropagation

The placeholders are set up with a lot more abstraction than DOOM, and is spread out in different methods under the `QLearning` class.
 
#### loss

##### DOOM

TD error simply getting the value and returning,these functions show good coding as it breaks it down enough for people unfamiliar with the code to understand and also there's alot of good commenting.


##### Cartpole

The code uses a higher level of extraction where the loss function is set up under the Qlearning class bt it essentiallly is the same. target Q-values = `q vals`.

#### fill feed dict 
##### Cartpole

Overdoing abstraction. A function to take action,rewards and state into a dictionary for processing.
`labels`=rewards

### Aggregating Layer
##### DOOM


`self.Q` equations is the aggregating layer and is used as you simply can't just add them together (duelling system). Instead we use the formula: `Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))`.

#### Absolute error

##### DOOM
```
self.absolute_errors = tf.abs(self.target_Q - self.Q)# for updating Sumtree
```
The absolute error is the TD error and is used for Prioritised Experience Replay, not for the duelling DQN. It updates the SumTree which also modifies the loss. 

##### Cartpole

prediction is q value from the behaviour network .Label is q target found using bellman.
Then find the error between the two.

## Pretraining

This is done to prepopulate memory with experiences.


## Action Choosing

### DOOM
Its called predict_action but that just bad naming.First we randomise a number then we use improved epsilon greedy strategy then use the same exploration/exploitation

### Cartpole

exploitation vs exploration method to find the action based on the target network.

## Merging Target and behraviour networks

### DOOM
Later it copies values from behav to target so it is our double DQN implementation. This function is simply initialising the parameters, this can also be done in the constuctor/init function.

#### Cartpole
copies all trainable variables of the behavioural network into the target network.

## Training

### DOOM

- initialise variables, parameters
- initialise game environment
- update target parameters with dqn weights
- for 5000 episodes
  - get state from screen, held in buffer
  - Stack the frames 4 at a time
  - for 5000 steps
    - increment step, tau, decay step
    - get s, a, r
    - if episode is finished
      - no next state
      - calculate total rewards
      - assign s,a,r,s' to experience variable 
      - store experience in memory
    - else
      - get s'
      - assign s,a,r,s' to experience variable 
      - store experience in memory
      - s = s'
    -Take our experience samples and assign each part to a variable array
    - run both behaviour and target networks with the inputs only being next states to find Q-values
    -for batch size
      - action from max Q values in behavioural network is found 
      - if at terminal state
        - append reward to target Q array
      - else
        - calculate target Q using Bellman equation
        - append target Q to target Q array
    - target Q array is reshaped
    - calculate loss and absolute errors by inputting states, target Q array, actions and Importance Sampling weights into the computational graph (Tensorflow)
    - priority values are updated in the memory Sumtree
    - update target values after 10000 steps using tau variable (fixed q values)

### Cartpole  

#### def Train step

Trains the network
if a summary object has not been created/running then calculate loss, predictions etc... else if there's a session running then continue feeding the tuples into it.


## playing

The game is played for 10 epsidoes where we don't update the q networks and also don't use expeience replay.

------------------------------------- notes 



## Main
```
if np.mean(last_100) >= 195:
                gym.upload('/tmp/cartpole-experiment-1',
                    api_key=sys.argv[1])
                exit()

```
If it succesfully completes the episode in the last 100 tries then it ends the algorithm


----------------------------------------

The hidden layer 



**** See [Prioritised Experience Replay markdown] ()