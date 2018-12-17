"""
format of code:
    
1. dependencies and parameters
2. initalisation of Q table
3. learning algorithm for training
    for every episode
        get start state
        for every step
            (show rendered env)
            choose action
            update Q table, rewards, state
4. use trained model to play game
"""

#better for small q table you can explore every single state
import numpy as np
import gym 
import random 
import time

env = gym.make('Taxi-v2')

# hyperparameters 
episode_max = 50000
steps_max = 99

learn_rate = 0.5
discount = 0.5

# exploration parameters
exploration_rate = 1.0  # epsilon
decay_rate = 0.01

# initialise Q table
Q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Learning Q-algorithm
for episode in range(episode_max):

    state = env.reset()
    done = 0

    for step in range(steps_max):

        print('*************** START TRAINING ******************')
        env.render()
        time.sleep(0.5)

        # choose action 
        # [Numpy] random: mathematical operation choosing random from uniform distribution, values are equally likely to be chosen
        xx_tradeoff = random.uniform(0, 1)

        if xx_tradeoff > exploration_rate:
            # exploitation
            action = np.argmax(Q_table[state,:])
        else:
            # exploration
            # [OpenAI Gym] sample: picking one value from array of values
            action = env.action_space.sample()

    state_, reward, done, info = env.step(action)

    # update Q table using algorithm
    Q_table[state,action] += learn_rate * (reward + discount * (np.max(Q_table[state_, :]) - Q_table[state, action]))

    state = state_

    if done:
        break


# Testing policy on game environment, do not update Q-table anymore
# environment still changes: new pick-off/drop-off
cumulative_return = []

for runs in range(100):
    state = env.reset()
    total_steps =0
    total_rewards = 0

    for steps in range(99):
        env.render()

        # greedy method only
        action = np.argmax(Q_table[state,:])

        state_, reward, done, info = env.step(action)

        total_rewards += reward

        state = state_

        if done: 
            cumulative_return.append(total_rewards)
            break

env.close()

print('Score: ', str(sum(cumulative_return)/episode_max))


