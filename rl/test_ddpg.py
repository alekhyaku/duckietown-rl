import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from ddpg import DDPG
from learning.utils.wrappers import  ResizeWrapper

# Initialize the Duckietown environment
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)

# Initialize the DDPG agent
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
max_action = float(env.action_space.high[0])
print("Initializing the DPPG agent")
agent = DDPG(action_dim, max_action)
print("Done with DDPG")

agent.load(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model/")

obs = env.reset()
done = False

while True:
    while not done:
        action = agent.predict(obs)
        # Perform action
        print("action: ", action)
        obs, reward, done, _ = env.step(action)
        env.render()
    done = False
    obs = env.reset()

