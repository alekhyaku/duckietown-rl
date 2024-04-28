import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper
from ddpg import DuckieRewardWrapper
import numpy as np
import os
import os.path
from space_wrapper import DiscreteWrapper

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Define the size of the output from the CNN
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(160, kernel_sizes[0], strides[0]), kernel_sizes[1], strides[1]), kernel_sizes[2], strides[2])
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(120, kernel_sizes[0], strides[0]), kernel_sizes[1], strides[1]), kernel_sizes[2], strides[2])

        linear_input_size = convw * convh * 64

        # Define the fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)

        self.action_layer = nn.Linear(512, action_dim)
        self.value_layer = nn.Linear(512, 1)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        # Pass the input through the CNN
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the CNN
        x = x.contiguous().view(x.size(0), -1)

        # Pass the flattened output through the fully connected layers
        x = F.relu(self.fc1(x))

        # Compute action probabilities and state values
        action_probs = F.softmax(self.action_layer(x), dim=-1)
        state_values = self.value_layer(x)

        return action_probs, state_values

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action.squeeze(-1).long())
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def act(self, state, memory):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_probs, _ = self.forward(state)  
        m = torch.distributions.Categorical(action_probs)  
        action = m.sample()

        memory.states.append(state.squeeze(0))
        memory.actions.append(action)
        memory.logprobs.append(m)

        return action.item()

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

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
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Converting list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        tensor_logprobs = [torch.tensor(logprob.probs) for logprob in memory.logprobs]
        old_logprobs = torch.stack(tensor_logprobs).to(device).detach().squeeze(1)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            taken_old_logprobs = old_logprobs[torch.arange(len(old_actions)), old_actions]

            # Now taken_old_logprobs should have shape [50], and you can subtract it from logprobs:
            ratios = torch.exp(logprobs - taken_old_logprobs.detach())
                
            # Finding Surrogate Loss:
            # Convert all tensors to Float
            ratios = ratios.float()
            state_values = state_values.float()
            rewards = rewards.float()
            dist_entropy = dist_entropy.float()

            # Compute the loss
            advantages = rewards - state_values.detach()
            advantages = advantages.float()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss = loss.float()
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
    env = DiscreteWrapper(env)
    env = NormalizeWrapper(env)
    env = DuckieRewardWrapper(env, crash_coef=25)
    render = True
    solved_reward = 300         
    log_interval = 20          
    max_episodes = 1000        
    max_timesteps = 1000        

    update_timestep = 50      
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    lr = 0.0003
    betas = (0.9, 0.999)
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var=64, lr=0.002, betas=(0.9, 0.999), gamma=0.99, K_epochs=4, eps_clip=0.2)

    # load if model exists
    if os.path.exists('/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_Duckietown-udem1-v0.pth'): 
        ppo.policy.load_state_dict(torch.load('/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_Duckietown-udem1-v0.pth'))
    print(lr,betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # catch keyboard interrupts to save model
    try:
        # training loop
        for i_episode in range(1, max_episodes+1):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                # Running policy_old:
                action = ppo.policy_old.act(state, memory)
                state, reward, done, _ = env.step(action)

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                # Decay the crash coef every thirty steps
                if timestep % 30 == 0:
                    if env.crash_coef > 1:
                        env.crash_coef *= .50

                running_reward += reward
                if render:
                    env.render()
                if done:
                    break
            
            # save every episode reward to a csv file from a directory
            with open('/home/alekhyak/gym-duckietown/rl/train_rewards/ppodiscrete_rewards.csv', 'a') as f:
                f.write(str(running_reward) + '\n')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            avg_length += t

            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './PPODiscrete_{}.pth'.format(env_name))
                break
            
            # save the policy every 10 episodes in case of a crash
            if i_episode % 10 == 0:
                print("Training interrupted, about to save..")
                torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_{}.pth'.format(env_name))
                print("Finished saving..should return now!")  

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0

        print("Training interrupted, about to save..")
        torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_{}.pth'.format(env_name))
        print("Finished saving..should return now!")
        
    except KeyboardInterrupt:
        print("Training interrupted, about to save..")
        torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_{}.pth'.format(env_name))
        print("Finished saving..should return now!")

if __name__ == '__main__':
    main()