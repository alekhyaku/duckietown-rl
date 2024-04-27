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
from learning.utils.wrappers import ResizeWrapper, NormalizeWrapper
from gym import spaces
import os
import os.path
import csv
from ddpg import DuckieRewardWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

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
        self.fc1 = nn.Linear(11*16*64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0  # Normalize the input images
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, num_actions, gamma=0.9, learning_rate=0.00025, buffer_size=10000, target_update=10):
        self.net = DQN(num_actions)
        self.target_net = DQN(num_actions)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        self.target_update = target_update
        self.steps_done = 0
        self.gamma = gamma
        
    def select_action(self, state, action_dim, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(state)
            return q_values.max(1)[1].item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)
        # Compute Q values and loss, and update the main network
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q values
        current_q_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network every self.target_update steps
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    # Add an experience to the replay buffer
    def push_experience(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def save_reward(save, filename, directory, reward):
        with open("{}/{}.csv".format(directory, filename), "a") as f:
            writer = csv.writer(f)
            writer.writerow([reward])

    def load(self, filename, directory):
        self.net.load_state_dict(
            torch.load("{}/{}.pth".format(directory, filename), map_location=device)
        )
    
    def save(self, filename, directory):
        print("Saving to {}/{}.pth".format(directory, filename))
        torch.save(self.net.state_dict(), "{}/{}.pth".format(directory, filename))
        print("Saved Actor")


    # Sample a batch of experiences from the replay buffer and use them to train the network
    def optimize(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        state      = torch.FloatTensor(np.array(state))
        next_state = torch.FloatTensor(np.array(next_state))
        action     = torch.LongTensor(np.array(action))
        reward     = torch.FloatTensor(np.array(reward))
        done       = torch.FloatTensor(np.array(done))

        # Compute the Q-values for the current states and the next states
        current_q_values = self.net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]

        # Compute target Q values
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network every self.target_update steps
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())

if __name__ == "__main__":
    # Number of episodes to train for
    num_episodes = 1000

    # Number of steps to take in each episode
    num_steps = 1000

    # Batch size for network updates
    batch_size = 16

    # Initialize the environment and the agent
    env = gym.make("Duckietown-udem1-v0")
    env = ResizeWrapper(env)
    env = DiscreteWrapper(env)
    env = NormalizeWrapper(env)
    env = DuckieRewardWrapper(env, crash_coef=25)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    agent = DQNAgent(action_dim)

    # load previously trained model to train more
    if os.path.exists("/home/alekhyak/gym-duckietown/rl/model/dqn.pth"):
        agent.load("dqn", "/home/alekhyak/gym-duckietown/rl/model")
        print("loaded agent")

    # we want to catch keyboard interrupts to save the model
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(num_steps):
                # Select an action
                action = agent.select_action(state, action_dim)

                # Take a step in the environment
                next_state, reward, done, _ = env.step(action)

                # Store the experience in the replay buffer
                agent.push_experience(state, action, reward, next_state, done)

                # Update the network
                agent.optimize(batch_size)

                # Decay the noise standard deviation every thirty steps
                if step % 30 == 0:
                    if env.crash_coef > 1:
                        env.crash_coef *= .50

                # Update the current state and episode reward
                state = next_state
                episode_reward += reward
                # env.render()
                if done:
                    break

            print(f"Episode {episode}: {episode_reward}")
            agent.save_reward("dqn", "/home/alekhyak/gym-duckietown/rl/train_rewards", episode_reward)
            if episode % 10 == 0:
                print("10 episodes done, saving model")
                agent.save(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model")
                print("Finished saving, back to work...")

        print("Training done, about to save..")
        agent.save(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model")
        print("Finished saving..should return now!")

    except KeyboardInterrupt:
        print("Training done, about to save..")
        agent.save(filename="dqn", directory="/home/alekhyak/gym-duckietown/rl/model")
        print("Finished saving..should return now!")   