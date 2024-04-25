import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('rewards.csv', header=None)

# Load your rewards here
ddpg_rewards = [...]  
baseline_rewards = [...]

# Plot DDPG rewards
plt.plot(ddpg_rewards, label='DDPG')

# Plot baseline rewards
plt.plot(baseline_rewards, label='Baseline')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.show()