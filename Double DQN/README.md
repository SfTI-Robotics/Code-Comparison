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
Morvan uses the `n_features*2+2` again in this code as it quite flexible. You times two for next state then plus two for reward and action. This forms the eperience transition sample.


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
def train_model(self):
       
        # array of current Q-value, input is current state 
        target = self.model.predict(update_input)

        # an array of next Q-values, input next state
        target_next = self.model.predict(update_target)

        # an array of Q-values 
        #fixed q target network
        #target val is when the target model is updated to become the evaluator model
        # input is next state
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                # decoupling
                a = np.argmax(target_next[i])
                # Bellman equation
                #q value only stored ine target network not e
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        # calculates loss and does optimisation 
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

```
There's 3 q values caluculated in this the first two is q and q' using the behavioural network(being feed minibatches).Then we use the target network get the q value from target network. For each batch the q value  is updated based on the q value from the target  network run through the bellman equation but the action a is chosen using q' in behavioural network.Last;y the model is fitted using target network(needs to evaluate policy in target as well).
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
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next(Target Network Q-value) depending on above actions
else:
        selected_q_next = np.max(q_next, axis=1)    # the natural DQN

# Bellman equation
q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

_, self.cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: batch_memory[:, :self.n_features],
                                        self.q_target: q_target})
self.cost_his.append(self.cost)
```
The Q-values are updated by using the behavioural network to select an action. The Q-value from the target network (`selected_q_next`) is used in the Bellman equation, changing the behavioural Q-value (`q_target`) and then is stored in the target network.