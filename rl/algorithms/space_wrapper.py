import gym
import numpy as np
from gym import spaces

class DuckieRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, crash_coef):
        super(DuckieRewardWrapper, self).__init__(env)
        self.crash_coef = crash_coef

    def reward(self, reward):
        if reward == -1000:
            reward = -10*self.crash_coef
        elif reward < 0:
            reward *= .25
        return reward
    
class DiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(6)
        print(self.action_space)

    def action(self, action):
        # Turn left and forward
        if action == 0:
            vels = [0.6, +1.0]
        # Turn right and forward
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        # Go left
        elif action == 3:
            vels = [0.0, 0.5]
        # Go right
        elif action == 4:
            vels = [0.0, -0.5]
        # brake
        elif action == 5:
            vels = [0.0, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)