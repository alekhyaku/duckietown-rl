import ast
import argparse
import logging
import gym
import os
import numpy as np
from test_ppo_discrete import SaveReturn

# Duckietown Specific
from rl.algorithms.ddpg import DDPG, DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper

# Initialize the Duckietown environment
env = gym.make("Duckietown-small_loop-v0")
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
crash_coef = 25
env = DuckieRewardWrapper(env, crash_coef)

# Initialize the DDPG agent
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
max_action = float(env.action_space.high[0])
print("Initializing the DPPG agent")
agent = DDPG(action_dim, max_action, action_space=env.action_space)
print("Done with DDPG")

agent.load(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model/")
saved = SaveReturn("/rl/test_return/", "ddpg_return.csv")
obs = env.reset()
done = False
episode_num = 0

while True:
    while not done:
        action = agent.select_action(obs, False)
        # Perform action
        print("action: ", action)
        obs, reward, done, _ = env.step(action)
        # record the reward for the episode
        total_reward += reward
        env.render()
    done = False
    episode_num +=1
    # Save this episode and reward for it
    saved.save(episode_num, total_reward)
    obs = env.reset()

