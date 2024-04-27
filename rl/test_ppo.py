import ast
import argparse
import logging
import gym
import os
import numpy as np
import torch

# Duckietown Specific
from ppo import PPO, Memory
from ddpg import DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper

class SaveReturn:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
    def save(self, episode, reward):
        self.data.append((episode, reward))
        with open(self.filename, 'w') as f:
            for episode, reward in self.data:
                f.write(f"{episode}, {reward}\n")


saved = SaveReturn("ppo__return.csv")
 # Initialize the environment and the agent
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = DuckieRewardWrapper(env, 1)
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
print("Initializing the PPO agent")
agent = PPO(state_dim, action_dim)
memory = Memory()

agent.policy.load_state_dict(torch.load('/home/alekhyak/gym-duckietown/rl/model/PPO_Duckietown-udem1-v0.pth'))

obs = env.reset()
done = False
total_reward = 0
episode_num = 0
while True:
    while not done:
        action = agent.policy.act(obs, memory)
        # Perform action
        obs, reward, done, _ = env.step(action)
        # record the reward for the episode
        total_reward += reward
        env.render()
    done = False
    episode_num +=1
    # Save this episode and reward for it
    saved.save(episode_num, total_reward)
    obs = env.reset()
