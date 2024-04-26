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
from dqn import DiscreteWrapper

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

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(11*16*64, 512)
        self.out = nn.Linear(512, action_dim)

    def forward(self, state):
        if state.dim() == 4:
            state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)  # flatten the tensor
        print("X: ", x.shape)
        x = F.relu(self.fc(x))
        logits = self.out(x)
        prob = F.softmax(logits, dim=-1)
        print("Action probability: ,", prob)
        return F.softmax(logits, dim=-1)
    
    def evaluate(self, state, action):
        action_probs = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        print("Action logprobs", action_logprobs.shape)
        dist_entropy = dist.entropy()

        return action_logprobs, state, torch.squeeze(dist_entropy)

    def act(self, state, memory):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = torch.distributions.Categorical(probs)
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
        
        self.policy = PolicyNetwork(action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = PolicyNetwork(action_dim).to(device)
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
        # Convert Categorical objects to Tensor objects
        tensor_logprobs = [torch.tensor(logprob.probs) for logprob in memory.logprobs]
        print(old_states.shape, old_actions.shape, len(tensor_logprobs))
        # Then stack them
        old_logprobs = torch.stack(tensor_logprobs).to(device).detach().squeeze(1)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            print(logprobs.shape, state_values.shape, dist_entropy.shape)
            print(old_logprobs.shape, old_logprobs)
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
    env_name = "Duckietown-zigzag_dists"
    env = gym.make(env_name)
    env = ResizeWrapper(env)
    env = DiscreteWrapper(env)
    env = NormalizeWrapper(env)
    env = DuckieRewardWrapper(env, crash_coef=25)
    render = True
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode

    update_timestep = 2000      # update policy every n timesteps
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    # max_action = float(env.action_space.high[0])
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
    ppo = PPO(state_dim, action_dim, n_latent_var=64, lr=0.002, betas=(0.9, 0.999), gamma=0.99, K_epochs=4, eps_clip=0.2)
    # load if model exists
    if os.path.exists('/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_Duckietown-udem1-v0.pth'): 
        ppo.policy.load_state_dict(torch.load('/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_Duckietown-udem1-v0.pth'))
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
            
            #save every episode reward to a csv file from a directory
            with open('/home/alekhyak/gym-duckietown/rl/rewards/ppodiscrete_rewards.csv', 'a') as f:
                f.write(str(running_reward) + '\n')
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            avg_length += t

            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './PPODiscrete_{}.pth'.format(env_name))
                break

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0
    except KeyboardInterrupt:
        print("Training interrupted, about to save..")
        torch.save(ppo.policy.state_dict(), '/home/alekhyak/gym-duckietown/rl/model/PPODiscrete_{}.pth'.format(env_name))
        print("Finished saving..should return now!")

if __name__ == '__main__':
    main()