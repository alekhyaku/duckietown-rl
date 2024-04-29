# A Comparative Analysis of Discrete versus Continuous Action Spaces in  Reinforcement Learning for Autonomous Driving Simulation in Duckietown

## Introduction
This Repository uses source code from two open source github repositiories
1. Gym-Duckietown
    * [https://github.com/duckietown/gym-duckietown]
2. AIDO RL baseline
    * [https://github.com/duckietown/challenge-aido_LF-baseline-RL-sim-pytorch]
    * This is used as the baseline to compare our implementations against, and copied over to the rl/baselines folder
    * We believe this baseline is created from the model.py file in this repository

This repository contains code for a final project for a Reinforcement Learning class at UT Austin. All code produced for this project can be found in the rl folder.

We tried to implement Deep Reinforcement Learning algorithms to solve in lane following problem in the Duckietown simulator. We trained three algorithms DDPG, DQN, and PPO but used both a discrete and continuous action space for PPO.

Our goal was to compare how defining the action space can affect performance when solving the lane following problem. We used CNN's for image processing, and kept the same reward function and state space to train all 4. 

## Installation
In order to generate reproducible results, this is the installation process we underwent in order to get gym-duckietown running, as well as to test our algorithms.

Everything was run on an Ubuntu 22.10 VM allocated 11100 MB of memory with 4 processors.

1. Have python 3.8 installed  
2. Have pip and git installed through the terminal
3. Clone this repository
4. Install the gym-duckietown requirements
    ```
    pip3 install -e .
    ```
5. Install Pillow
    ```
    pip install -U Pillow
    ```
6. Install torch
    ```
    pip install torch
    ```
7. Reinstall numpy version 1.22.4
    ```
    pip uninstall numpy
    pip install numpy==1.22.4
    ```
8. Reinstall pyglet with version 1.5.11
    ```
    pip uninstall pyglet
    pip install pyglet==1.5.11
    ```
9. export the python path 
    ```
     export PYTHONPATH="$/home/user/duckietown-rl/"
    ```

You can also check your installation against the requirements.txt file in the duckietown-rl/rl folder

## Running Code
### To train our algorithms:
1. cd into the rl/algorithms folder
2. run python3 dqn.py
    * Or ddpg.py, ppo,py, and ppo_dicrete.py
    ```
    cd rl/algoritnms
    python3 dqn.py
    ```
### To test our algorithms
*** The file paths will have to be changed in each test file for local installations of the repository ***
1. cd into the rl/testing folder
2. Make sure there is a .pth file in the /rl/model folder for the algorithm you want to test
3. Run python3 test_{name}.py
    * There is a test file for every algorithm and to run the baseline test
    * Each test file will create a {algorithm}_{map}_seed{seed}_return.csv file found in the test_return folder
    * To test all run the test_all file
    * To visualize the average return for the number of episodes tested for all tests, run python3 plot.py in the plots folder
    * Running the test files again will overwrite the existing returns created for our report
    ```
    cd rl/testing
    python3 test_dqn.py
    python3 test_all.py
    cd ../plots
    python3 plot.py
    ```
