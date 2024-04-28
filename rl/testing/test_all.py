import random
from test_ddpg import run_ddpg
from test_dqn import run_dqn
from test_ppo import run_ppo
from test_ppo_discrete import run_ppo_discrete
from test_baseline import run_baseline

# generate three random seeds
seeds = [random.randint(0, 1000) for _ in range(3)]
# save the names of the three maps to test on
udem1 = "Duckietown-udem1-v0"
small_loop = "Duckietown-small_loop-v0"
zigzag = "Duckietown-zigzag_dists-v0"
maps = [udem1, small_loop, zigzag]

# run the tests for the four algorithms on the three maps using the three seeds
# 100 episodes each
for seed in seeds:
    print("Running tests for seed: ", seed)
    for map in maps:
        print("Running tests for map: ", map)
        run_ddpg(env_name=map, seed=seed, max_episode_steps=100)
        run_dqn(env_name=map, seed=seed, max_episode_steps=100)
        run_ppo(env_name=map, seed=seed, max_episode_steps=100)
        run_ppo_discrete(env_name=map, seed=seed, max_episode_steps=100)
        run_baseline(env_name=map, seed=seed, max_episode_steps=100)