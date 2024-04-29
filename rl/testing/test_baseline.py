import gym
import numpy as np

# Assuming the baseline was created by the ddpg agent created by gym-duckietown
from rl.baseline.ddpg import DDPG
from rl.baseline.gym_wrapper import DTPytorchWrapper
from rl.algorithms.space_wrapper import DuckieRewardWrapper
from learning.utils.wrappers import  ResizeWrapper, NormalizeWrapper, ImgWrapper, ActionWrapper
from save_return import SaveReturn

def run_baseline(env_name="Duckietown-udem1-v0", seed=0, max_episode_steps=100):
    saved = SaveReturn("/home/alekhyak/gym-duckietown/rl/test_return/", f"baseline_{env_name}_seed{seed}_return.csv")
    # Initialize the Duckietown environment
    # Using the same reward function as our DDPG agent for fairness
    env = gym.make(env_name)
    env.seed(seed)
    image_processor = DTPytorchWrapper()
    crash_coef = 25
    env = DuckieRewardWrapper(env, crash_coef)

    # Initialize policy
    policy = DDPG(state_dim=image_processor.shape, action_dim=2, max_action=1, net_type="cnn")
    policy.current_image = np.zeros((640, 480, 3))
    policy.load(filename="model", directory="/home/alekhyak/gym-duckietown/rl/baseline/")
    obs = env.reset()
    done = False
    episode_num = 0
    # only test for max episodes
    while episode_num < max_episode_steps:
        total_reward = 0
        steps = 0
        while not done:
            action = policy.predict(image_processor.preprocess(obs))
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
    run_baseline()