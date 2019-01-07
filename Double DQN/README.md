# DDQN Code Comaprison

Cart Pole | [Code](https://github.com/simoninithomas/reinforcement-learning-1/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py)

Pendulum | [Code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5.1_Double_DQN)


## Overall differences
2 types
1)

behavioural network -> behaviour then sends to target
target chooses q value


### Diff 1 : Keras vs Tensorflow
Using Keras simplifies the computation set up process the Keras functions combine multiple Tensorflow functions into one. 

### Diff 2 : Abstraction/encapsulation
More functions to avoid repetition of code.

## in depth comparison

### Learning algorithm 

#### Memory

##### Cart Pole
The memory is set up using a deque data structure with a maximum capacity of 2000. 

Hyperparameters are hard-coded as instance variables only (instead of also set as constructor inputs). This Keras simplicity removes the amount of flexibility of the constructor. 

Set weights are used instead of randomised weights.


##### Pendulum 
Morvan uses the n_features*2+2 again in this code as it quite flexible. You times two for next state then plus two for reward and action. These will always be need no matter what environment.


### Building models
The target and evaluation (behavioural) model is built similar to DQN.

### Update target model
We update the target model more frequently instead of behaviour(dqns) 

#### Cart Pole

```
def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

```
Using the Keras function `set_weights()`, the weights are updated in the .h5 file. This is updated after every episode It causes overestimation, updating alot at beginning you get alot of bad q values but do get better towards end.This uses more computational power.

#### Pendulum 
```
# in init()
self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


# in learn()
if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
```
The parameters of the target model are updated through using a `replace_target_op` function. 
Initially epsidoe finishes very quickly but as it improves it updates over fewer episodes.This isnt required for ddqns but is an optimisation.

### Choosing Action
The action is chosen based on the behavioural model using the target Q-value.
#### Cart Pole


#### Pendulum 

### Experience Replay
#### Cart Pole


#### Pendulum 

