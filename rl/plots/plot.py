import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rl.testing.test_all import seeds, maps

# List of algorithms, maps, and seeds
algorithms = ['ddpg', 'dqn', 'ppo', 'ppo_discrete', 'baseline']

# Load the data from the CSV files
dataframes = []
for algorithm in algorithms:
    for map_name in maps:
        for seed in seeds:
            df = pd.read_csv(f'{algorithm}_{map_name}_seed{seed}_return.csv', header=None, names=['Episode', 'Reward'])
            df['Algorithm'] = algorithm
            df['Map'] = map_name
            df['Seed'] = seed
            dataframes.append(df)

# Concatenate all the dataframes into one
data = pd.concat(dataframes)

# Create a separate plot for each seed and map
for seed in seeds:
    for map_name in maps:
        # Subset the data for this seed and map
        subset = data[(data['Seed'] == seed) & (data['Map'] == map_name)]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='Episode', y='Reward', hue='Algorithm', ci='sd')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward per Episode (Seed: {seed}, Map: {map_name})')
        plt.legend(title='Algorithm')
        
        # Save the plot to a file
        plt.savefig(f'plot_seed_{seed}_map_{map_name}.png')
        
        plt.show()