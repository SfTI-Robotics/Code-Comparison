"""
Code does not have objects or functions
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adadelta, Adam
from time import sleep
import numpy as np
import random
import gym

# Environment
env = gym.make('CartPole-v1').env
inputCount = env.observation_space.shape[0]
actionsCount = env.action_space.n
 
# Neural Network
model = Sequential()
model.add(Dense(24, input_dim=inputCount, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actionsCount, activation='linear'))
model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

# Load weights from file, preset weights
model.load_weights("weights.h5")

# Hyperparameters
gamma = 1.0
epsilon = 1.0
epsilonMin = 0.01
epsilonDecay = 0.999
episodes = 5000

# Memory (Remember & Replay)
memory = []
batch_size = 64
memoryMax = 50000

# Training
for e in range(episodes):
    #  get initial state
    s = env.reset()
    s = np.array([s])

    for time in range(500):
        # Act greedy sometimes
        if np.random.rand() <= epsilon:
            a = random.randrange(actionsCount)
        else:
            a = np.argmax(model.predict(s))

        newS, r, done, _ = env.step(a)
        newS = np.array([newS])
        # q-value for current step
        target = r + gamma * np.max(model.predict(newS))
        print('target =', target)
        # target_f is a 2D matrix which consists of an action and q-value, batch of state
        # Question
        target_f = model.predict(s)[0]
        print('step')
        print(target_f)
        target_f[a] = target
         # fit(x, y): trains the model for a given number of iterations (epochs) on a data set
        model.fit(s, target_f.reshape(-1, actionsCount), epochs=1, verbose=0)
        memory.append((s, a, r, newS, done))
        s = newS

        # free first items in memory
        if len(memory)==memoryMax:
            del memory[:5000]

        if done:
            print("episode: {}/{}, score: {}".format(e, episodes, time))
            break

    # update exploration rate
    if epsilon > epsilonMin:
        epsilon *= epsilonDecay

    # Replay memory
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + gamma * np.max(model.predict(next_state))

            target_f = model.predict(state)[0]
            target_f[action] = target
            model.fit(state, target_f.reshape(-1, actionsCount), epochs=1, verbose=0)


# Save weights
model.save_weights("weights.h5")

# Play game
print("\nPlaying Game...")
sleep(1)

# testing 
s = env.reset()
done = False
while not done:
    env.render()
    a = np.argmax(model.predict(np.array([s])))
    newS, r, done, _ = env.step(a)
    s = newS