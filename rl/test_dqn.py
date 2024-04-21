import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from dqn import DQNAgent, DiscreteWrapper
from learning.utils.wrappers import  ResizeWrapper

 # Initialize the environment and the agent
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)
env = DiscreteWrapper(env)
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
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

