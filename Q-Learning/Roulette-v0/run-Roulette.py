
import gym
import numpy as np
import os
import time
import datetime
import sys

# from /home/lemon740/Gym-T4-Testbed import Roulette
#this is importing the algorithm which is defined as a class from a diffeerent file
# make sure file names do not contain '-', or else it will cause errors
from Roulette_brain import Q_Learning
from summary import * 


# Graphing results
now = datetime.datetime.now()

graph = summary(summary_types = ['sumiz_step', 'sumiz_reward', 'sumiz_epsilon'], 
            # the optimal step count of the optimal policy 
            step_goal = 0, 
            # the maximum reward for the optimal policy
            reward_goal = 0, 
            # maximum exploitation value
            epsilon_goal = 0.99,
            # desired name for file
            NAME = "Roulette-v0-" + str(now), 
            # file path to save graph. i.e "/Desktop/Py/Scenario_Comparasion/Maze/Model/"
            SAVE_PATH = "/Gym-T4-Testbed/"
    )

#our bash file takes the arg.txt file that has a list of the different environments episodes and steps
#when using sys.argv start by using 1 not 0 for the first agrument because 0 will be the python file name
EPISODE_NUM = int(sys.argv[2])
STEP_NUM = int(sys.argv[3])

env = gym.make(sys.argv[1])
#action space=38
#state =1
actions=env.action_space.n
states=env.observation_space.n
# the class doesn't take in any parameters but to pass in the action and state space in your class
#  use the name __init__ with double underscores
QLearning = Q_Learning(states, actions)

for episode in range(EPISODE_NUM):

    episode_rewards = 0
    observation = env.reset()

    start_time = time.time()

    for step in range(STEP_NUM):

        # os.system('clear')
        # time.sleep(0.1)

        # roulete doesn't have env.render

        #each step you choose action by calling the choose action function in our algorithm file
        action = QLearning.choose_action(observation, EPISODE_NUM, env) # 

        # apply action
        observation_, reward, done, _ = env.step(action)

        #learn from the action(updating q values)
        QLearning.learn(observation, action, reward, observation_, done)
        #get cumalative reward
        episode_rewards += reward
        #move on to next state
        observation = observation_

        if done:
            # print('Episode =', episode, ' step =', step,  'reward =', episode_rewards)
            break
    
    graph.summarize(episode, step, time.time() - start_time, episode_rewards, QLearning.epsilon)
        
print(QLearning.q_table)     
print('game over')
env.close()

            







