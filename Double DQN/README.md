# DDQN Code Comaprison

Cart Pole | [Code](https://github.com/simoninithomas/reinforcement-learning-1/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py)

Pendulum | [Code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5.1_Double_DQN)


## Overall differences

### Diff 1 : Keras vs Tensorflow
Using Keras simplifies the computation set up process the Keras functions combine multiple Tensorflow functions into one. 

### Diff 2 : Abstraction/encapsulation
More functions to avoid repetition of code.

## in depth comparison

### Learning algorithm 
-how memory is set up
-build model

#### Memory

##### Cart Pole
The memory is set up using a deque data structure with a maximum capacity of 2000. 

Hyperparameters are hard-coded as instance variables only (instead of also set as constructor inputs). This Keras simplicity removes the amount of flexibility of the constructor. 

Set weights are used instead of randomised weights.


##### Pendulum 
Morvan uses the n_features*2+2 again in this code as it quite flexible. You times two for next state then plus two for reward and action. These will always be need no matter what environment.


### Building models
The target and evaluation (behavioural) model is built.

### Learning from experience 
#### Cart Pole


#### Pendulum 

### Choosing Action
#### Cart Pole


#### Pendulum 