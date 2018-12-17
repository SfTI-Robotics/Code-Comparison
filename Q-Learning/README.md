# Q-learning

### Frozen lake 
Arthur Juliani [Article](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) | [Code](https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb)

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