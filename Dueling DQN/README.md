# Duelling DQN Code Comaprison

Cart Pole | [Code]()

Doom [Article](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)
 | [Code](https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb)


## Overall differences

### Diff 1 : Keras vs Tensorflow
Using Keras simplifies the computation set up process the Keras functions combine multiple Tensorflow functions into one. 

### Diff 2 : Abstraction/encapsulation
More functions to avoid repetition of code.

## In depth comparison

### Create Environment

creating object and environment

#### Parameters

- action space = identity matrix
```
possible_actions = np.identity(7,dtype=int).tolist()
    
    return game, possible_actions


game, possible_actions = create_environment()
```
Use this when choosing actions
```
state_size = [100,120,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels) 
action_size = game.get_available_buttons_size() 

```
7 buttons may also be pause button but only need 5.

```
state_size = [100,120,4] 
```
We have 4 channels because of the 4 frames stacked ont top of eachother.


### Memory
#### Frames
Only every 4 frames is considered because having only 1 frame doesn't allow our agent to decide the motion of the objects in the game


We use deque to stack the frames every episode.First apending the frame on the deque and then stack the array of frames onto eachother(4 dimensions). 

```
stacked_state = np.stack(stacked_frames, axis=2)
```

We consider 2 stacks(axis=2) one with the initial state and one with the next state(each with 4 frames).

If full remove the oldest one(smalest q values and least reliable).
Each time we have a new state deque removes the next state frames and first.

## DDDQN Algorithm

### initialisation
Uses tensorflow .
Has 4placeholders that will be used later on.

### Conv NN
So it uses 3 convulutional layers. Each one first using the con2d function to split the frame into parts eg:player, victims etc... 

then we use elu to scale it down

### Building Model

Here we build two NN one for the value function and the second advantage function.The inputs for both are the flattened processed layers discussed above.Then the second layers for each take the previous layer and give the output V(s) is the reward number and advantage is the action to take. this is where the dueling DQN is implemented.


## Sumtrees
Binary Trees which have a property that the children nodes have to add to form the parent node.

They store the priority values of the experience transitions. 

## Memory

### Intialisation
e,a,b is used for importance sampling see kevins paper
Its initialised as a sum tree but also an array for our experience 

### Storing
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



### Experience Replay
sample: 
first it creates the two empty arrays

Is Weights increases to avoid earlier experiences later on(as the are inaccurate/not reliable)

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
copying values from behav to target so it is our double DQN implementation 


## Training

fixed q values thing: 
update target values after 10000 steps using tau variable




Questions 
- what are filters
- isweights
- - sumtrees
