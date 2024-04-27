import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from rl.algorithms.ppo_discrete import PPO, Memory
from rl.algorithms.dqn import DiscreteWrapper
from rl.algorithms.ddpg import DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper

class SaveReturn:
    def __init__(self, directory, filename):
        self.filename = os.path.join(directory, filename)
        self.data = []
    def save(self, episode, reward):
        self.data.append((episode, reward))
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w') as f:
            for episode, reward in self.data:
                f.write(f"{episode}, {reward}\n")

saved = SaveReturn("/rl/test_return/", "ppo_discrete_return.csv")
 # Initialize the environment and the agent
env = gym.make("Duckietown-zigzag_dists")
env = ResizeWrapper(env)
env = DiscreteWrapper(env)
env = NormalizeWrapper(env)
env = DuckieRewardWrapper(env, 1)
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
print("Initializing the PPO agent")
agent = PPO(state_dim, action_dim,n_latent_var=64, lr=0.002, betas=(0.9, 0.999), gamma=0.99, K_epochs=4, eps_clip=0.2)
memory = Memory()
# load if model exists
agent.policy.load(filename="PPODiscrete_Duckietown-zigzag_dists", directory="/home/alekhyak/gym-duckietown/rl/model/")

obs = env.reset()
done = False
total_reward = 0
episode_num = 0
while True:
    while not done:
        action = agent.policy.act(obs)
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
