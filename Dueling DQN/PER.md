# Prioritised Experience Replay

We use the same code as duelling dqns but split the two methods to show clearly  how tbwy are implemented

Cart Pole | [Code]()

Doom [Article](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)
 | [Code](https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb)


Really important to use PER when using double and duelling since we can't just use a replay buffer. We don't choose experiences randomly but rather select them based on priorisation.

## DOOM

## Sumtrees
Binary Trees which have a property that the children nodes have to add to form the parent node. They store the priority values of the experience transitions. Leaf Nodes are special nodes, that are at the bottom tier of tree.

### def  init: 

 Initialize our SumTree data object with all nodes = 0 and data (data array) with all = 0. The tree capacity ( = number of nodes) is calculated by taking the leaf node number (`capacity`) multiplied by 2 and subtract 1. `Data` array store experience of leaf nodes.

### def  add: 

add our priority score in the sumtree leaf and experience (S, A, R, S', Done) in data. The `tree index` is the left-most leaf node's index. `data_pointer` is an index for the data array. 

 ### def  update: 
 we update the leaf priority score and propagate through tree. The old priority score is replaced by the new priority score. Then we trace/traverse back along the branch to the upper tiers of the Sum Tree. These nodes are incremented with the difference between the new and old priority score.
 
A priority score is a value assigned to each node showing that importance of that experience(ie:this action might lead to a very high reward(1,000,000 pts)). How ever each priority also has a probability (ie: the million reward has a 1/1000 chance of getting it).
 
 ### def  get_leaf: 

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
 
 ### def  total_priority: 
 
 get the root node value to calculate the total priority score of our replay buffer. 



## Memory

We no longer use deques as when it adds and removes it changes the indexes for every experiences(not efficient)
 

### def  init: 

generates our sumtree and data by instantiating the SumTree object. The parameters `e,a,b` are used for importance sampling. The memory object (`self.tree`) is initialised a sum tree and also a data array for our experiences. 

![alt text](https://cdn-images-1.medium.com/max/1600/0*0qPwzal3qBIP0eFb)

our first parameter(e) is used to calculate the priority score along with td error


Lastly in b represents beta in the importance sampling equation

![alt text](https://cdn-images-1.medium.com/max/1400/0*Lf3KBrOdyBYcOVqB)


### def  store: 

we store a new experience in our tree. Each new experience will have priority = max_priority (and then this priority will be corrected during the training (when we'll calculating the TD error hence the priority score). 

```
max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
# If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
# So we use a minimum priority
if max_priority == 0:
    max_priority = self.absolute_error_upper

self.tree.add(max_priority, experience)   # set the max p for new p
```
The maximum priority score is found from the memory SumTree. A safety precaution is taken when the max priority is found to be 0. A min priority is needed for this. Then the max priority is assigned to the inputted experience in the SumTree using the `add` method. 

### def  sample:

#### DOOM

1. First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.   
The batch size (`n`) is inputted into the function. The batch index (`b_idx`) and Importance Sampling weights (`b_ISWeights`)  variables are initialised as empty arrays. Then we split the experiences into n range. The priority segments (ranges) are found by the total priority divided by the batch size. The importance sampling bias variable (`b`) is incremented when a new minibatch of experience is sampled.

2. Then a value is uniformly sampled from each range
A for loop is used to iterate through the batch size. From each priority segment (range), a value is randomly sampled between a maximum and minimum (variables `a` and `b`). 

3. We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
Using the `get_leaf` function, the sumTree index, priority score, and experience is returned. 

4. Then, we calculate IS weights for each minibatch element
   The sampling probabilities is dividing the priority score by the total priority score and raised to the power of 1. This ensures that only the highest priority is selected. 

![alt text](https://cdn-images-1.medium.com/max/1400/0*iCkLY7L3R3mWEh_O)

The Importance sampling weights is used to correct the bias introduced by the frequently-appearing experience samples. Is Weights increases to avoid earlier experiences later on(as the are inaccurate/not reliable). The weights are updated in the formula below, which is a rearrangement of the equation in the figure. The weights are normalised for stability by dividing by the maximum weight.
```
p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
max_weight = (p_min * n) ** (-self.PER_b)

b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
```
![alt text](https://cdn-images-1.medium.com/max/1400/0*Lf3KBrOdyBYcOVqB)

#### Cart pole

```
segment = self.tree.total() / BATCH_SIZE
for i in range(BATCH_SIZE):
    minimum = segment * i
    maximum = segment * (i+1)
    s = random.uniform(minimum, maximum)
    (idx, p, data) = self.tree.get(s)
    batch.append((idx, data))
return batch
```

The priority segment (ranges) is calculated by dividing the total priority by 10 (batch size). Using a for loop, we iterate throught the 10 ranges and randomly select a value (`s`) within the upper and lower bounds of each range. 

### def  update_batch: 

Update the priorities on the tree. The first line is the same as the priority value equation mentioned before but then it makes sure abs errors is within bounds.Then we add a randomness factor to choosing the priority score by making it to the power of parameter a(0.6) This allows us to also choose some more random samples instead of just high priority scores(as they may not be the best options to take). Update Sumtree priority by using the `update` function. 

--------------------------------------------
## Cart pole

## Sumtrees

## replay Memoy
### Add
`(error + MEMORY_BIAS)`is just the priority(pt) where as the priority variable is a probability of that priority


### sample
create segments and randomly select a sample to append using   `get` function from SumTree

### def train with batch

The target network is updated when after 50 episodes (fixed q target) using the `q_net.copy_to` function. We select a batch of experiences for experience replay. Since the SumTree object is created in a separate python file and imported, its functions are inherited. 
`Labels` is used throughout the code to represent different things in this function it represents the q vqlue this is a good example of BAD NAMING.

