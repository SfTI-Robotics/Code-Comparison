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

Here we build two NN one for the value function and the second advantage function.The inputs for both are the flattened processed layers discussed above.Then the second layers for each take the previous layer and give the output V(s) is the reward number and advantage is the action to take.


## Sumtrees
Binary Trees which have a property that the children nodes have to add to form the parent node.


## Memory

### Intialisation
e,a,b is used for importance sampling see kevins paper

### Storing
```
max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

```

### Experience Replay

### Updating


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
## Training



Questions 
- what are filters
- isweights
- - sumtrees
