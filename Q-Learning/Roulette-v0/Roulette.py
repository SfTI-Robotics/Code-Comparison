import numpy as np
import random

#used to decay epsilon after each step
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE=0.005

#good idea to do to get same "random" values
np.random.seed(1)

#here is the class we use to call our functions, it doesn't take in parameters
class QLearning:
#to make sure initialisation works the naming has to be __init__ withdouble underscoreson each side
    def __init__(self, states, actions):

        self.actions=actions
        self.states=states
        #generating q tables you can use pandas dataframe or just an array of zeros its a personal prefereence
        self.q_table = np.zeros((states, actions))
        self.alpha = 0.85
        self.gamma = 0.99
        self.epsilon = 1.0
    
    # e-greedy method for selecting actions 
    def choose_action(self, state, episode, env):
                            
        if np.random.rand() > self.epsilon :
            # exploration: randomly choose action
            action = env.action_space.sample()
        else :
            # exploitation: choose maximum action            
            action = np.argmax(self.q_table[state,:])

        # decay epsilon as episodes increase there's lots of different ways to decay it but we got this
        #equation from thomas simonini on free code camp
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
        return action
        
    # updates the Q-table for a specific state using the Bellman equation            
    def learn(self, state, action, reward, state_, done):
            
        if not done:
            q_target = self.gamma * (reward + np.max(self.q_table[state_,:]) )

        # no next state, episode finished        
        else: 
            q_target = reward

        q_predict = self.q_table[state, action]

        self.q_table[state,action] += self.alpha * (q_target - q_predict )



