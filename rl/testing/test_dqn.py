import gym
import numpy as np

# Duckietown Specific
from rl.algorithms.dqn import DQNAgent
from rl.algorithms.space_wrapper import DuckieRewardWrapper, DiscreteWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper
from save_return import SaveReturn

def run_dqn(env_name="Duckietown-udem1-v0", seed=0, max_episode_steps=100):
    saved = SaveReturn("/home/alekhyak/gym-duckietown/rl/test_return/", f"dqn_{env_name}_seed{seed}_return.csv")
    # Initialize the environment and the agent
    env = gym.make(env_name)
    env.seed(seed)
    env = ResizeWrapper(env)
    env = DiscreteWrapper(env)
    env = NormalizeWrapper(env)
    env = DuckieRewardWrapper(env, 25)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    print("Initializing the DQN agent")
    agent = DQNAgent(action_dim)
    print("Done with DQN")

    agent.load(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model/")

    obs = env.reset()
    done = False
    episode_num = 0

    while episode_num < max_episode_steps:
        total_reward = 0
        steps = 0
        while not done:
            action = agent.select_action(obs, action_dim)
            # Perform action
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
    run_dqn()