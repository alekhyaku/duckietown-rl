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
from gym import spaces
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper
from learning.utils.env import launch_env
import os
import os.path
import csv

class DuckieRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, crash_coef):
        super(DuckieRewardWrapper, self).__init__(env)
        self.crash_coef = crash_coef

    def reward(self, reward):
        if reward == -1000:
            reward = -10*self.crash_coef
        else:
            reward *= .5
        return reward

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(64 * 13 * 18, 512)  # updated input size
        self.fc2 = nn.Linear(512, action_dim)
        self.max_action = max_action
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0,3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("Action before it gets clipped: ", x)
        # relu_val = self.sigm(abs(x[:, 0]))
        # print("Relu val: ", relu_val)
        x0 = self.max_action * self.sigm(abs(x[:, 0]))
        x1 = torch.tanh(x[:, 1])
        x = torch.stack((x0, x1), dim=1)
        # print("Action in actor forward: ", x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        # print("Action in actor after squeeze forward: ", x)
        # print("Action shape in actor forward: ", x.shape)
        return x

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size() + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state = state.permute(0,3, 1, 2)
        # Pass state through convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # print("Shape in forward", x.shape)
        # Flatten the output from conv layers
        x = x.contiguous().view(x.size(0), -1)
        
        # print("Shape in forward", x.shape)
        # print("ACtion shape in forward", action.shape)
        # print("Action in critic forward: ", action)
        # Concatenate the action to the feature vector
        if action.shape[1] == 1:
            action = action.squeeze(1)
        x = torch.cat([x, action], 1)
        
        # Pass the result through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # print("Final Critic value: ", x)
        
        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, 3, 120, 160)))).view(1, -1).size(1)

class DDPG(object):
    def __init__(self, action_dim, max_action, action_space, noise_std_dev=0.3, noise_decay=0.75):
        self.actor = Actor(action_dim, max_action)
        self.actor_target = Actor(action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(action_dim)
        self.critic_target = Critic(action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.noise_std_dev = noise_std_dev
        self.noise_decay = noise_decay   
        self.action_space = action_space

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), "{}/{}_actor.pth".format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), "{}/{}_critic.pth".format(directory, filename))
        print("Saved Critic")

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action= self.actor(state).cpu().data.numpy().flatten()
        # Add noise to the action
        noise = np.random.normal(0, self.noise_std_dev, size=action.shape)
        print("Noise: ", noise)
        action = action + noise

        # Clip the action to be within the valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # # Decay the noise standard deviation
        # self.noise_std_dev *= self.noise_decay
        # print("State ", state)
        # state = state.view(-1, 3, 480, 640)  # assuming state is your input and the image size is 480x640
        print("Action in select action: ", action)
        return action
    
    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
    def save_rewards(self, reward, filename, directory):
        with open("{}/{}.csv".format(directory, filename), "a") as f:
            writer = csv.writer(f)
            writer.writerow([reward])

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("{}/{}_actor.pth".format(directory, filename), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load("{}/{}_critic.pth".format(directory, filename), map_location=device)
        )
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            if len(replay_buffer) < batch_size:
                return
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            # print("D: ", d)
            # print("State ", x)
            # print("Action ", u)
            
            state = torch.FloatTensor(np.array(x)).to(device)
            action = torch.FloatTensor(np.array(u)).to(device)
            next_state = torch.FloatTensor(np.array(y)).to(device)
            done = torch.FloatTensor(np.array(1 - np.array(d))).to(device)
            reward = torch.FloatTensor(np.array(r)).to(device)

            # print("Action in train, ", action)
            # print("State Shape", state.shape)
            # print("Action before critic Shape", action.shape)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = target_Q.view(-1, 1)  # Ensure target_Q has shape [64, 1]      
            target_Q = reward.unsqueeze(1) + (done.unsqueeze(1) * discount * target_Q).detach()
            
            # Get current Q estimate
            current_Q = self.critic(state, action)

            # print("Target Q: ", target_Q)
            # print("Current Q: ", current_Q)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # print("Critic loss: ", critic_loss)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # print("Actor loss: ", actor_loss)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

if __name__ == "__main__":
    # Initialize the Duckietown environment
    env = gym.make("Duckietown-udem1-v0")
    # env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env.seed(0)

    # Initialize the DDPG agent
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    max_action = float(env.action_space.high[0])
    # print("max action ", max_action)
    print("Initializing the DPPG agent")
    agent = DDPG(action_dim, max_action, action_space=env.action_space)
    if os.path.isfile("/home/alekhyak/gym-duckietown/rl/model/ddpg_actor.pth"):
        print("Loading model")
        agent.load(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model/")
    print("Done with DDPG")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=10000)
    print("Initialized bufffer")

    num_episodes = 400  # number of training episodes
    num_steps = 1000  # number of steps per epoch
    batch_size = 16  # size of the batches to sample from the replay buffer
    discount = 0.99  # discount factor for the cumulative reward
    tau = 0.005  # target network update rate
    try:
        # Training loop
        for episode in range(num_episodes):
            agent.noise_std_dev = 0.3
            crash_coef = 25
            env = DuckieRewardWrapper(env, crash_coef)
            print("Episode ", episode)
            state = env.reset()
            env.seed(0)
            done = False
            steps = 0
            episode_reward = 0
            while steps < num_steps:
                action = agent.select_action(state)
                # print("Action in episode step ", steps, " : ", action)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, next_state, action, reward, done)
                state = next_state
                if steps%75 == 0:
                    agent.noise_std_dev *= agent.noise_decay
                # Decay the noise standard deviation every five steps
                if steps % 30 == 0:
                    crash_coef *= .50
                    env = DuckieRewardWrapper(env, crash_coef)
                steps += 1
                episode_reward += reward
                # print("Reward at step ", steps, " : ", reward)
                env.render()
                if done:
                    break

            agent.save_rewards(episode_reward, "ddpg", "/home/alekhyak/gym-duckietown/rl/rewards")
            print("Episode reward: ", episode_reward)
            print("about to train agent")
            # Train the agent
            agent.train(replay_buffer, iterations=batch_size, batch_size=batch_size, discount=discount, tau=tau)
            
            # save the policy every ten eipsodes in case something crashes
            if episode%10 == 0:
                print("Episode %10 done, about to save.. : ", episode)
                agent.save(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model")
                print("Finished saving..back to work")

        print("Training done, about to save..")
        agent.save(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model")
        print("Finished saving..should return now!")
    except KeyboardInterrupt:
        print("Training interrupted, about to save..")
        agent.save(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model")
        print("Finished saving..should return now!")