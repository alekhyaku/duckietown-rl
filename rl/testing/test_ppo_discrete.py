import ast
import argparse
import logging
import gym
import os
import numpy as np

# Duckietown Specific
from rl.algorithms.ppo_discrete import PPO, Memory
from rl.algorithms.space_wrapper import DiscreteWrapper, DuckieRewardWrapper
from learning.utils.wrappers import ResizeWrapper, NormalizeWrapper
from save_return import SaveReturn

def run_ppo_discrete(env_name="Duckietown-udem1-v0", seed=0, max_episode_steps=100):
    saved = SaveReturn("/rl/test_return/", f"ppo{env_name}_seed{seed}_discrete_return.csv")
    # Initialize the environment and the agent
    env = gym.make(env_name)
    env.seed(seed)
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
    agent.policy.load(filename="PPODiscrete_Duckietown-udem1-v0", directory="/home/alekhyak/gym-duckietown/rl/model/")

    obs = env.reset()
    done = False
    episode_num = 0
    while episode_num < max_episode_steps:
        total_reward = 0
        steps = 0
        while not done:
            action = agent.policy.act(obs, memory)
            # Perform action
            obs, reward, done, _ = env.step(action)
            # record the reward for the episode
            total_reward += reward
            steps += 1
            if steps % 30 == 0:
                if env.crash_coef > 1:
                    env.crash_coef *= .50
            # env.render()
        done = False
        episode_num +=1
        # Save this episode and reward for it
        saved.save(episode_num, total_reward)
        obs = env.reset()

if __name__ == "__main__":
    run_ppo_discrete()