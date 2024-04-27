import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from rl.algorithms.dqn import DQNAgent, DiscreteWrapper
from rl.algorithms.ddpg import DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper

 # Initialize the environment and the agent
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)
env = DiscreteWrapper(env)
env = NormalizeWrapper(env)
env = DuckieRewardWrapper(env, 1)
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
print("Initializing the DQN agent")
agent = DQNAgent(action_dim)
print("Done with DQN")

agent.load(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model/")

obs = env.reset()
env.seed(0)
done = False

while True:
    while not done:
        action = agent.predict(obs)
        # Perform action
        obs, reward, done, _ = env.step(action)
        env.render()
    done = False
    obs = env.reset()
