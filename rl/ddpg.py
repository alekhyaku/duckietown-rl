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
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
import os
import os.path


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
        # print("Action in forward: ", x)
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
        
        # Flatten the output from conv layers
        x = x.contiguous().view(x.size(0), -1)
        
        # print("Shape in forward", x.shape)
        # Concatenate the action to the feature vector
        x = torch.cat([x, action.squeeze(1)], 1)
        
        # Pass the result through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, 3, 120, 160)))).view(1, -1).size(1)

class DDPG(object):
    def __init__(self, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.actor_target = Actor(action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(action_dim)
        self.critic_target = Critic(action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
    
    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), "{}/{}_actor.pth".format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), "{}/{}_critic.pth".format(directory, filename))
        print("Saved Critic")

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        # print("State ", state)
        # state = state.view(-1, 3, 480, 640)  # assuming state is your input and the image size is 480x640
        return self.actor(state).cpu().data.numpy().flatten()
    
    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
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
            # print("Action Shape", action.shape)
            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = target_Q.view(-1, 1)  # Ensure target_Q has shape [64, 1]
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

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
    env = ResizeWrapper(env)

    # Initialize the DDPG agent
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    max_action = float(env.action_space.high[0])
    # print("max action ", max_action)
    print("Initializing the DPPG agent")
    agent = DDPG(action_dim, max_action)
    if os.path.isfile("/home/alekhyak/gym-duckietown/rl/model/ddpg_actor.pth"):
        print("Loading model")
        agent.load(filename="ddpg", directory="/home/alekhyak/gym-duckietown/rl/model/")
    print("Done with DDPG")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=10000)
    print("Initialized bufffer")

    num_episodes = 1000  # number of training episodes
    num_steps = 500  # number of steps per epoch
    batch_size = 8  # size of the batches to sample from the replay buffer
    discount = 0.99  # discount factor for the cumulative reward
    tau = 0.005  # target network update rate
    try:
        # Training loop
        for episode in range(num_episodes):
            print("Episode ", episode)
            state = env.reset()
            done = False
            steps = 0
            episode_reward = 0
            while steps < num_steps:
                action = agent.select_action(state)
                # print("Action in episode step ", steps, " : ", action)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, next_state, action, reward, done)
                state = next_state
                if done:
                    break
                steps += 1
                episode_reward += reward
                # env.render()

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