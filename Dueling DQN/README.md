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

#### Preprocessed

The pixels of a single frame are normalised so that each pixel has a similar distribution. this helps the stochastic gradient descent converges faster.

#### Stack frame

Only every 4 frames is considered because having only 1 frame doesn't give the neural network a sense of motion for the objects in the game.
We use deque to stack the frames every episode.First appending the frame on the deque and then stack the array of frames onto each other(4 dimensions). When full, the deque automatically removes the oldest one(smalest q values and least reliable).


```
stacked_state = np.stack(stacked_frames, axis=2)
```

We consider 2 stacks(axis=2) one with the initial state and one with the next state(each with 4 frames).

`stacked_state` is a stack data type and `stacked_frames` is a deque data type. 



## DDDQN Algorithm
### Init
#### DOOM
Has 4 placeholders that will be used later on: inputs, importance sampling weights, actions and target Q-value.

### Building Model
#### DOOM

##### Convolutional NN
So it uses 3 convolutional layers. Each one uses the `conv2d` function to split the frame into parts eg:player, victims etc... https://insiders.liveshare.vsengsaas.visualstudio.com/join?01F3A2AE9F04469D55627BF70C5046B81F12
Then we use elu network to restrict the data to a range and adds the layers.

#### NN
Here we build two separate NN, one for the value function and the second advantage function. The inputs for both are the flattened processed layers discussed above.Then the second layers for each take the previous layer and give the output .V(s) is the reward number and advantage is the action to take. This is where the dueling DQN is implemented.

##### Aggrigating Layer
`self.Q` equations is the aggrigating layer and is used as you simply can't just add them together (duelling system). Instead we use the formula: `Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))`.

#### Absolute error
```
self.absolute_errors = tf.abs(self.target_Q - self.Q)# for updating Sumtree
```
The absolute error is the TD error and is used for Prioritised Experience Replay, not for the duelling DQN. It updates the SumTree which also modifies the loss. 
            

## Sumtrees
Binary Trees which have a property that the children nodes have to add to form the parent node. They store the priority values of the experience transitions. Leaf Nodes are special nodes, that are at the bottom tier of tree.

def init: 

 Initialize our SumTree data object with all nodes = 0 and data (data array) with all = 0. The tree capacity ( = number of nodes) is calculated by taking the leaf node number (`capacity`) multiplied by 2 and subtract 1. `Data` array store experience of leaf nodes.

def add: 

add our priority score in the sumtree leaf and experience (S, A, R, S', Done) in data. The `tree index` is the left-most leaf node's index. `data_pointer` is an index for the data array. 

 def update: 
 we update the leaf priority score and propagate through tree. The old priority score is replaced by the new priority score. Then we trace/traverse back along the branch to the upper tiers of the Sum Tree. These nodes are incremented with the difference between the new and old priority score.
 
A priority score is a value assigned to each node showing that importance of that experience(ie:this action might lead to a very high reward(1,000,000 pts)). How ever each priority also has a probability (ie: the million reward has a 1/1000 chance of getting it).
 
 def get_leaf: 

 retrieve priority score(tree[leaf_index]), index and experience associated with a leaf(data[data_index]). 

_ We are going to use the example that thomas_ _puts in his code for this section where the index and priority score is the same for each node:_

             0  
            / \
          1     2
         / \   / \
        3   4 5   6  
 
 _We will take v=5 and the index we should expect is 5._

 In the while loop, the left child node value is compared to the random value input (`v`). If the value is smaller, the index will be replaced by the left child node index.

 Otherwise, the value is subtracted by the left child node value. The index will also be replaced by the right child node index.

  _Loop 1: Our example starts out with parent_index(p)=0, v =0, then left_child_index(l)=1, right_child_index(r)=2. Since l is less then the length of the tree(number of nodes) which is 7. Then we move on the scond if else statement where the first condition is false and v is bigger than the left child's priority score(5>3) so v is then changed to minus the right child priority score(2) from itself(5-2=3).And the parent index becomes(2)._
 
  Once the index reaches the bottom tier of the tree, the search is ended and the index is assigned to the variable `leaf_index`.


  _Loop 2: now our values are v=3, p=2, l=5, r=6. The fist If statement is false so we go to the second if else staement and the first condiotion is true(3<5) so p=5._

_Loop 3: v=3, p=5, l=11, r=12 The first if statement becomes true(11>7) so we have found our leaf index=5 and we break the loop and retur the values_  

 The data array index corresponds to the position where the experience (linked to the leaf nodes) is stored. We have found the leaf node which contains the priority score we want. To find the corresponding experience, we find the position the experience is stored in the data array. The `data_index` is calculated by taking the leaf node index subtracted by the number of leaf nodes (`capacity`) and adding 1. 
 
 def total_priority: 
 
 get the root node value to calculate the total priority score of our replay buffer. 



## Memory

We no longer use deques as when it adds and removes it changes the indexes for every experiences(not efficient)
 

def init: 

generates our sumtree and data by instantiating the SumTree object. The parameters `e,a,b` are used for importance sampling. The memory object (`self.tree`) is initialised a sum tree and also a data array for our experiences.
![alt text](https://cdn-images-1.medium.com/max/1600/0*0qPwzal3qBIP0eFb)

def store: 

we store a new experience in our tree. Each new experience will have priority = max_priority (and then this priority will be corrected during the training (when we'll calculating the TD error hence the priority score). 

The maximum priority score is found from the memory SumTree. A safety precaution is taken when the max priority is found to be 0. Then the max priority is assigned to the inputted experience in the SumTree using the `add` method. 

```
max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
# If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
# So we use a minimum priority
if max_priority == 0:
    max_priority = self.absolute_error_upper

self.tree.add(max_priority, experience)   # set the max p for new p

```
Store memory=ies in tree, looks a t leaves and gets the max. Does this by extracting the tree value data (tree) that is stored in the SumTree (self.tree). #badnaming
A min priority is needed incase our environment is exatcly how it needs to be(failsafe).

Updates the tree to add the new expeirnce instead of sum tree. assigns the data (experience) into the date frame. " we store a new experience in our tree. Each new experience will have priority = max_priority (and then this priority will be corrected during the training (when we'll calculating the TD error hence the priority score)."

new experiences are equal to max priority



def sample:

1. First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.   
The batch size (`n`) is inputted into the function. The batch index (`b_idx`) and Importance Sampling weights (`b_ISWeights`)  variables are initialised as empty arrays. The priority segments (ranges) are found by the total priority divided by the batch size. 
The importance sampling bias variable (`b`) is incremented when a new minibatch of experience is sampled.

2. Then a value is uniformly sampled from each range
A for loop is used to iterate through the batch size. From each priority segment (range), a value is randomly sampled between a maximum and minimum (variables `a` and `b`). 

3. We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
Using the `get_leaf` function, the sumTree index, priority score, and experience is returned. 

4. Then, we calculate IS weights for each minibatch element


def update_batch: 

update the priorities on the tree



### Prioritized Experience Replay

Rally important to use when using double and duelling we can't just use a replay buffer, we don't choose them randomly but rather select them based on priorisation score. Binary Tree is used for efficiency.
sample: 
first it creates the two empty arrays

Is Weights increases to avoid earlier experiences later on(as the are inaccurate/not reliable)

Then we split the experiences into n range. And we get a random value for each range and using this value we select an experience corresponding to that value(get leaf).

`b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight`

We then ge the importance sampling weight but also divide by max weights to make sure it wont be bigger than 1.

Increase value of b as we have more samples from the minibatch 

minimum priority is divided by total priority to find find the smallest chance of using an experience. 

`max_weight = (p_min * n) ** (-self.PER_b)`
probablity is inversely proportional
the minimum priority is multiplied by the minibatch sample size, then raised to the negative bias  value.

A for loop is used to search the SumTree for the randomly selected value from the minibatch of experiences. This uses the `get_leaf` function. a random value (probability?) is generated and the corresponding priority value, experience and index of priority value in tree.

Only sampling between two sets(range is quite small)


### Updating
Here we increas abs error until it reaches 1 and takes the min between it and 1.
update Sumtree priority by calculating the priority score from error raised to the power of the a parameter, then using the `update` function. 

## Learning


## Action Choosing
```
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation) 
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability
```
First we randomise a number then we use improved epsilon greedy strategy then use the same explortion/explitation

## update target graph
copying values from behav to target so it is our double DQN implementation then we calclate q target of next state(Double DQN)




## Training

fixed q values thing: 
update target values after 10000 steps using tau variable

Fisrt we use the DQN to get q values for necxt state then 




Questions 
- what are filters
- isweights
- - sumtrees

# Second code

## Building model

value layers only have num of states to 1 as each state can only have one value but advantage is always num of states to num of actions as there's different q values
 
## error and calculations
we use the formula value+(advantage-error) to solve backpropagattion



## Qlearning

### init 
initialising everything
predictions-is the q predictioncs
` self.q_vals = self.q_vals(self.predictions, self.actions_pl)`
this is simply decalring that q_vals is a function 

## q_vals 
simply getting the value and returning

## loss

TD error simply getting the value and returning,these functions show good coding as it breaks it down enough for people unfamiliar with the code to understand and also there's alot of good commenting.
Logits=q vals

## Training

Returning a method for training the network using an optimiser and mininmising the loss.
global step counter is updated

### fill feed dict 
Overdoing abstraction
A function to take action,rewards and state into a dictionary for processing.
labels=rewards

### Train step
Traing the network
if a summary object has not been created/running then calculate loss, predictions etc... else if there's a session running then continue feeding the tuples into it.

### predict


### save_Tranisition

Putting tuples into the dictionary and adding it into its replay memory

# transition
save transition into a usable format in your replay buffer

### Select action
exploitation vs exploration= target prediction
predictions= values use max q ro get action

### train with batch

fixed q target so the behav network becomes the targte network blah blah
select batch of experiences for experience replay
Labels is used throughout the code to represent different things in this function it represents the q vqlue this is a good example of BAD NAMING.


### calcError

prediction is q behaviour 

label is q target found using bellman

then find thge error between the two

## replay Memoy
### Add
`(error + MEMORY_BIAS)`is just the priority(pt) where as the priority variable is a probability of that priority


### sample
create segments and randomly select a sample to append using   `get` function from SumTree

## Main
```
if np.mean(last_100) >= 195:
                gym.upload('/tmp/cartpole-experiment-1',
                    api_key=sys.argv[1])
                exit()

```
If it succesfully completes the episode in the last 100 tries then it ends the algorithm


#### Cart Pole

The hidden layer 
