import numpy as np 
import gym
import keras
import math
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Perameters #
num_episodes = 1000
num_to_win = 199
max_env_steps = None

discount_factor = 0.995
learning_rate = 0.01
learning_rate_decay = 0.01

epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.005

batch_size = 64
monitor = False
quiet = False

memory = deque(maxlen=50000)
env = gym.make('CartPole-v0')
if max_env_steps is not None:
    env._max_episode_steps = max_env_steps

# NN Model #
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=learning_rate_decay))

# Defining Necessary Function #
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def chose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random_sample() <= epsilon) else np.argmax(model.predict(state)[0])

def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1- math.log10((t+1)*epsilon_decay)))

def preprocess_state(state):
    return np.reshape(state, [1,4])

def replay(batch_size, epsilon):
    x_batch = []
    y_batch = []
    minibatch = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + discount_factor*np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=5, verbose=0)

    if epsilon > epsilon_min:
        epsilon = epsilon*epsilon_decay

# Defining Run Funcion#
def run():
    scores = deque(maxlen=25)

    for e in range(num_episodes):
        state = preprocess_state(env.reset())
        done = False
        i=0
        while not done:
            action = chose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i = i+1

        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= num_to_win:
            #model.save_weights("model.h5")
            if not quiet:
                print(f'Solved after {e} episodes')

        if e % 25 == 0 and not quiet:
            print(f'Episode {e} mean survival over last 25 episodes is {mean_score}')

        replay(batch_size, get_epsilon(e))

    if not quiet:
        print('Did not solve after {num_episodes} episodes')

    return e


run()