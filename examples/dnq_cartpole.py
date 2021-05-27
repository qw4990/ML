import gym
import random
import torch
import torch.nn as nn
import math
from dqn import DQN

# Observation:
#     Type: Box(4)
#     Num     Observation               Min                     Max
#     0       Cart Position             -4.8                    4.8
#     1       Cart Velocity             -Inf                    Inf
#     2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
#     3       Pole Angular Velocity     -Inf                    Inf
# Actions:
#     Type: Discrete(2)
#     Num   Action
#     0     Push cart to the left
#     1     Push cart to the right
# Reward:
#     Reward is 1 for every step taken, including the termination step
env = gym.make('CartPole-v0')
state_dim = 4
action_dim = 2


dnq = DQN(4, 2)
for i_epi in range(80):
    state0 = env.reset()
    for i_round in range(200):
        action = dnq.act(state0)
        state1, reward, done, info = env.step(action)
        if done:
            reward = -10
        
        dnq.learn(state0, action, state1, reward)
        
        if done:
            print("training episode {} finish after {} steps".format(i_epi, i_round))
            break
        state0 = state1

for i_test in range(10):
    state0 = env.reset()
    for i_round in range(200):
        action = dnq.act(state0)
        state1, _, done, _ = env.step(action)
        if done:
            print("testing episode {} finish after {} steps".format(i_test, i_round))
            break
        state0 = state1