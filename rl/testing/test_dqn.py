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
from test_ppo_discrete import SaveReturn

saved = SaveReturn("/rl/test_return/", "dqn_return.csv")
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
episode_num = 0

while True:
    while not done:
        action = agent.predict(obs)
        # Perform action
        obs, reward, done, _ = env.step(action)
        e# record the reward for the episode
        total_reward += reward
        env.render()
    done = False
    episode_num +=1
    # Save this episode and reward for it
    saved.save(episode_num, total_reward)
    obs = env.reset()
