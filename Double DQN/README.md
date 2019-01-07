# DDQN Code Comaprison

Cart Pole | [Code](https://github.com/simoninithomas/reinforcement-learning-1/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py)

Pendulum | [Code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/5.1_Double_DQN)


## Overall differences
2 types
1)

behavioural network -> behaviour(action) then sends to target
target chooses q value and runs a policy on i, and chooses policy
2)
another network with different parameter set
fixed q value determines using para and updates network using that 

q 

b-choose action
 - determine q network
t- chooses optimal policy

fixed 




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
```
# get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])
```
He uses self.model to choose instead of self.target_model as he did in dqns.

#### Pendulum 
```
# choose action based on Q-value from evaluation network
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
```
The computational graph calculates the Q-value from the evaluation network instead of the target network.

### Experience Replay and update Q-value
#### Cart Pole
```
# get action for the current state and go one step in environment
action = agent.get_action(state)
next_state, reward, done, info = env.step(action)
next_state = np.reshape(next_state, [1, state_size])

# if an action make the episode end, then gives penalty of -100
reward = reward if not done or score == 499 else -100

# save the sample <s, a, r, s'> to the replay memory
agent.append_sample(state, action, reward, next_state, done)

# every time step do the training
agent.train_model()
score += reward
state = next_state
```
The transition samples are stored by appending them in to the memory deque. 

#### Pendulum 
```
# select random experience
if self.memory_counter > self.memory_size:
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
else:
        sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
batch_memory = self.memory[sample_index, :]

# compute Q-values
q_next, q_eval4next = self.sess.run(
        [self.q_next, self.q_eval],
        feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                self.s: batch_memory[:, -self.n_features:]})    # next observation
q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

q_target = q_eval.copy()

# extract information from transition samples
batch_index = np.arange(self.batch_size, dtype=np.int32)
eval_act_index = batch_memory[:, self.n_features].astype(int)
reward = batch_memory[:, self.n_features + 1]

if self.double_q:
        # decoupling the action taken and the Q-value selected 
        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval(Evaluator Network Q-Value)
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next(Target Network Q - value) depending on above actions
else:
        selected_q_next = np.max(q_next, axis=1)    # the natural DQN

# Bellman equation
q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

_, self.cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: batch_memory[:, :self.n_features],
                                        self.q_target: q_target})
self.cost_his.append(self.cost)
```
