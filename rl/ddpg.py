import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from collections import namedtuple, deque
import random
import numpy as np
import gym_duckietown
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

# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # First Critic Network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # Concatenate state and action
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)  # No activation function, because output is Q-value
        return x

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Convert batches to tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device)

        print("Next state batch", next_state_batch)
        # print("shape", next_state_batch.shape)

        # Compute the target Q value
        target_Q = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        target_Q = reward_batch + (1 - done_batch) * self.discount_factor * target_Q.detach()

        # Compute the current Q value
        current_Q = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = self.critic_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            # Sample a batch of transitions from the replay buffer
            batch = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(np.array(batch.state)).to(device)
            action = torch.FloatTensor(np.array(batch.action)).to(device)
            next_state = torch.FloatTensor(np.array(batch.next_state)).to(device)
            reward = torch.FloatTensor(np.array(batch.reward)).to(device)
            done = torch.FloatTensor(np.array(1 - batch.done)).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state.cpu().numpy()))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get the current Q estimate
            current_Q = self.critic(state, action)

            # Compute the critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute the actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        

# Initialize the Duckietown environment
env = gym.make("Duckietown-udem1-v0")

# Define hyperparameters
num_episodes = 1000
batch_size = 64

# Initialize the DDPG agent
state_dim = np.prod(env.observation_space.shape)
action_dim = np.prod(env.action_space.shape)
max_action = float(env.action_space.high[0])
agent = DDPGAgent(state_dim, action_dim, max_action)
print("State dim", state_dim)
print("Action dim", action_dim)

# Initialize the replay buffer
replay_buffer = ReplayBuffer(max_size=1000000)

# Main training loop
for episode in range(1, num_episodes + 1):
    print("Episode num ", episode)
    obs = env.reset()
    done = False
    while not done:
        # Select action
        print("Observation", obs)
        print(obs.shape)
        action = agent.select_action(np.array(obs))
        # Execute action
        new_obs, reward, done, _ = env.step(action)
        # Store transition in the replay buffer
        replay_buffer.push(obs, action, reward, new_obs, done)
        obs = new_obs

        # Train the agent
        if len(replay_buffer) >= batch_size:
            # Sample a batch of transitions
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
            # Convert batches to tensors and move them to the appropriate device
            # state_batch = torch.FloatTensor(np.array(state_batch)).to(device)            
            # action_batch = torch.FloatTensor(np.array(action_batch)).to(device)
            # reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
            # next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
            # done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

            print("Next state batch", next_state_batch)
            # print("shape ", next_state_batch.shape)

            # Update the Critic and Actor networks
            agent.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)