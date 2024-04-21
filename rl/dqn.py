import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from collections import namedtuple, deque
import random
import numpy as np
import gym_duckietown
import gym
from collections import deque
import random
from learning.utils.wrappers import ResizeWrapper
from gym import spaces

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0.6, +1.0]
        # Turn right
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.float() / 255.0  # Normalize the input images
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, num_actions, learning_rate=0.00025, buffer_size=10000):
        self.net = DQN(num_actions)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state)
            return q_values.max(1)[1].item()

    # Add an experience to the replay buffer
    def push_experience(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # Sample a batch of experiences from the replay buffer and use them to train the network
    def optimize(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        state      = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action     = torch.LongTensor(action)
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(done)

        # Compute the Q-values for the current states and the next states
        current_q_values = self.net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.net(next_state).max(1)[0]

        # Compute the target Q-values
        target_q_values = reward + 0.99 * next_q_values * (1 - done)

        # Compute the loss between the computed Q-values and the target Q-values
        loss = F.mse_loss(current_q_values, target_q_values.detach())

        # Update the network parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Number of episodes to train for
num_episodes = 1000

# Number of steps to take in each episode
num_steps = 1000

# Batch size for network updates
batch_size = 8

# Initialize the environment and the agent
env = gym.make("Duckietown-udem1-v0")
env = ResizeWrapper(env)
env = DiscreteWrapper(env)
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
agent = DQNAgent(action_dim)

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(num_steps):
        # Select an action
        action = agent.select_action(state)

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)

        # Store the experience in the replay buffer
        agent.push_experience(state, action, reward, next_state, done)

        # Update the network
        agent.optimize(batch_size)

        # Update the current state and episode reward
        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode}: {episode_reward}")