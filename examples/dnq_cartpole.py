import gym
import random
import torch
import torch.nn as nn
import math
from dqn import DQN

# Description:
#     A pole is attached by an un-actuated joint to a cart, which moves along
#     a frictionless track. The pendulum starts upright, and the goal is to
#     prevent it from falling over by increasing and reducing the cart's
#     velocity.
# Source:
#     This environment corresponds to the version of the cart-pole problem
#     described by Barto, Sutton, and Anderson
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
#     Note: The amount the velocity that is reduced or increased is not
#     fixed; it depends on the angle the pole is pointing. This is because
#     the center of gravity of the pole increases the amount of energy needed
#     to move the cart underneath it
# Reward:
#     Reward is 1 for every step taken, including the termination step
# Starting State:
#     All observations are assigned a uniform random value in [-0.05..0.05]
# Episode Termination:
#     Pole Angle is more than 12 degrees.
#     Cart Position is more than 2.4 (center of the cart reaches the edge of
#     the display).
#     Episode length is greater than 200.
#     Solved Requirements:
#     Considered solved when the average return is greater than or equal to
#     195.0 over 100 consecutive trials.
env = gym.make('CartPole-v0')
dnq = DQN(4, 2)
for i_epi in range(80):
    state0 = env.reset()
    for i_round in range(200):
        action = dnq.act(state0)
        state1, reward, done, info = env.step(action)

        # TODO: adjust the reward according to the pole angle

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