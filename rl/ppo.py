import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import gym
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper
from ddpg import DuckieRewardWrapper
import numpy as np
import os
import os.path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, hidden_dim, num_outputs, initial_std=1.0, decay_factor=0.99):
        super(ActorCritic, self).__init__()

        # Define the CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Define the size of the output from the CNN
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        # Adjust these values to match your actual kernel sizes and strides
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(160, kernel_sizes[0], strides[0]), kernel_sizes[1], strides[1]), kernel_sizes[2], strides[2])
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(120, kernel_sizes[0], strides[0]), kernel_sizes[1], strides[1]), kernel_sizes[2], strides[2])

        linear_input_size = convw * convh * 64

        # Define the fully connected layers
        self.fc1 = nn.Linear(linear_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

        # Add a standard deviation attribute
        self.std = initial_std
        self.decay_factor = decay_factor

    def decay_std(self):
        # Decay the standard deviation after each episode
        self.std *= self.decay_factor

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Pass the input through the CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # print(x.size())

        # Flatten the output from the CNN
        x = x.contiguous().view(x.size(0), -1)

        # Pass the flattened output through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # print(x)

        # Apply sigmoid to the first element of x (velocity) and scale by 0.8
        x[0, 0] = torch.sigmoid(x[0, 0]) * 0.8

        # Apply tanh to the second element of x (steering)
        x[0, 1] = torch.tanh(x[0, 1])

        # x is a single action value
        mean = x

        # Use a decaying standard deviation
        std = torch.tensor(self.std)

        # Create a normal distribution over actions
        dist = torch.distributions.Normal(mean, std)

        return dist
    
    def act(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.forward(state)
        # print(dist)
        action = dist.sample()
        action[0, 0] = action[0, 0].clamp(0, 0.8)
        action[0, 1] = action[0, 1].clamp(-1, 1)
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        # print("action: ", action)
        
        return action.cpu().data.numpy().flatten()

# Define the PPO algorithm
class PPO:
    def __init__(self, num_inputs, num_outputs, hidden_size=256, lr=3e-4, betas=(0.9, 0.999), gamma=0.99, eps_clip=0.2, K_epochs=80):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(hidden_size, num_outputs).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(hidden_size, num_outputs).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    # Update policy by taking a single step of gradient descent
    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(int(self.K_epochs)):
            # Evaluating old actions and values :
            dist, state_values = self.policy(old_states)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

# Train the model
def main():
    ############## Hyperparameters ##############
    env_name = "Duckietown-udem1-v0"
    env = gym.make(env_name)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = DuckieRewardWrapper(env, crash_coef=25)
    render = True
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 1000        # max timesteps in one episode

    update_timestep = 50      # update policy every n timesteps
    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    max_action = float(env.action_space.high[0])
    hidden_dim = 256
    lr = 0.0003
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip)
    # load if model exists
    if os.path.exists('/home/alekhyak/gym-duckietown/rl/model/PPO_Duckietown-udem1-v0.pth'): 
        ppo.policy.load_state_dict(torch.load('/home/alekhyak/gym-duckietown/rl/model/PPO_Duckietown-udem1-v0.pth'))
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    try:
        # training loop
        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            # print(state.shape)
            # print("State: ", state)
            for t in range(max_timesteps):
                timestep += 1

                # Running policy_old:
                action = ppo.policy_old.act(state, memory)
                print("action: ", action)
                state, reward, done, _ = env.step(action)
                print("reward: ", reward)
                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                # Decay the noise standard deviation every thirty steps
                if timestep % 30 == 0:
                    if env.crash_coef > 1:
                        env.crash_coef *= .50

                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
                
            ppo.policy.decay_std()
            #save every episode reward to a csv file from a directory
            with open('/home/alekhyak/gym-duckietown/rl/rewards/ppo_rewards.csv', 'a') as f:
                f.write(str(running_reward) + '\n')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            avg_length += t

            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
                break
            
            if i_episode % 10 == 0:
                print("Training interrupted, about to save..")
                torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPO_{}.pth'.format(env_name))
                print("Finished saving..should return now!")

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0
        print("Training interrupted, about to save..")
        torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPO_{}.pth'.format(env_name))
        print("Finished saving..should return now!")
    except KeyboardInterrupt:
        print("Training interrupted, about to save..")
        torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPO_{}.pth'.format(env_name))
        print("Finished saving..should return now!")

if __name__ == '__main__':
    main()