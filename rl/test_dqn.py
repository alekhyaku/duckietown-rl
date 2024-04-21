import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from dqn import DQNAgent
from learning.utils.wrappers import  ResizeWrapper

# Initialize the Duckietown environment
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)

# Initialize the DDPG agent
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
max_action = float(env.action_space.high[0])
print("Initializing the DQN agent")
agent = DQNAgent(action_dim)
print("Done with DQN")

agent.load(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model/")

obs = env.reset()
done = False

while True:
    while not done:
        action = agent.predict(obs)
        # Perform action
        obs, reward, done, _ = env.step(action)
        env.render()
    done = False
    obs = env.reset()

