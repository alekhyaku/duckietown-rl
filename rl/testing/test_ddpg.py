import gym
import numpy as np

# Duckietown Specific
from rl.algorithms.ddpg import DDPG
from rl.algorithms.space_wrapper import DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper
from save_return import SaveReturn

def run_ddpg(env_name="Duckietown-udem1-v0", seed=0, max_episode_steps=100):
    saved = SaveReturn("/home/alekhyak/gym-duckietown/rl/test_return/", f"ddpg_{env_name}_seed{seed}_return.csv")
    # Initialize the Duckietown environment
    env = gym.make(env_name)
    env.seed(seed)
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
    obs = env.reset()
    done = False
    episode_num = 0
    # only test for max episodes
    while episode_num < max_episode_steps:
        total_reward = 0
        steps = 0
        while not done:
            action = agent.select_action(obs, False)
            # Perform action
            # print("action: ", action)
            obs, reward, done, _ = env.step(action)
            # record the reward for the episode
            total_reward += reward
            # Decay the crash coef every thirty steps
            if steps % 30 == 0:
                if env.crash_coef > 1:
                    env.crash_coef *= .50
            steps += 1
            # env.render()
        done = False
        episode_num +=1
        # Save this episode and reward for it
        saved.save(episode_num, total_reward)
        obs = env.reset()

if __name__ == "__main__":
    run_ddpg()