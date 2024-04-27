# A Comparative Analysis of Discrete versus Continuous Action Spaces in  Reinforcement Learning for Autonomous Driving Simulation in Duckietown

## Introduction
This Repository uses code from two other open source github repositiories
1. Gym-Duckietown
    * [https://github.com/duckietown/gym-duckietown]
2. AIDO RL baseline
    * [https://github.com/duckietown/challenge-aido_LF-baseline-RL-sim-pytorch]
    * This is used as the baseline to compare our implmentations against, and copied over to the rl/baselines folder

This repository contains code for a final project for a Reinforcement Learning class at UT Austin. All code produced for this project can be found in the rl folder.

We tried to implement Deep Reinforcement Learning algorithms to solve in lane following problem in the Duckietown simulator. We trained three algorithms DDPG, DQN, and PPO but used both a discrete and continuous action space for PPO.

Our goal was to compare how defining the action space can either hinder or help solve the lane following problem. We used CNN's for image processing, and kept the same reward function and state space to train all 4. 

## Installation
In order to generate reproducible results, this is the installation process we underwent in order to get gym-duckietown running, as well as to test our algorithms.

Everything was run on an Ubuntu 22.10 VM allocated 11100 MB of memory with 4 processors.

1. Have python 3.8 installed  
2. Have pip and git installed through the terminal
3. Clone this repository
4. pip3 install -e .
5. pip install -U Pillow
6. Install torch
7. pip uninstall numpy
8. pip install numpy==1.22.4
9. pip uninstall pyglet
10. pip install pyglet==1.5.11
11. export the python path 
    * export PYTHONPATH="$/home/user/gym-duckietown/"

You can also check your installation against the requirements.txt file in the gym-duckietown/rl folder

## Running Code
To train our algorithms:
1. cd into the rl folder
2. run python3 dqn.py
    * Or ddpg.py, ppo,py, and ppo_dicrete.py
To test our algorithms
1. cd into the rl folder
2. Make sure there is a .pth file in the /rl/model folder for the algorithm you want to test
3. Run python3 test_{name}.py
    * There is a test file for every algorithm and to run the baseline test
    * Each test file will create a {name}_return.csv file
    * To visualize the average return for the number of episodes tested, run python3 plot.py