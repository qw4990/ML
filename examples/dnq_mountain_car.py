import gym
import random
import torch
import torch.nn as nn
import math

# Description:
#     The agent (a car) is started at the bottom of a valley. For any given
#     state the agent may choose to accelerate to the left, right or cease
#     any acceleration.
# Source:
#     The environment appeared first in Andrew Moore's PhD Thesis (1990).
# Observation:
#     Type: Box(2)
#     Num    Observation               Min            Max
#     0      Car Position              -1.2           0.6
#     1      Car Velocity              -0.07          0.07
# Actions:
#     Type: Discrete(3)
#     Num    Action
#     0      Accelerate to the Left
#     1      Don't accelerate
#     2      Accelerate to the Right
#     Note: This does not affect the amount of velocity affected by the
#     gravitational pull acting on the car.
# Reward:
#      Reward of 0 is awarded if the agent reached the flag (position = 0.5)
#      on top of the mountain.
#      Reward of -1 is awarded if the position of the agent is less than 0.5.
# Starting State:
#      The position of the car is assigned a uniform random value in
#      [-0.6 , -0.4].
#      The starting velocity of the car is always assigned to 0.
# Episode Termination:
#      The car position is more than 0.5
#      Episode length is greater than 200
env = gym.make('MountainCar-v0')
dqn = DQN(2, 3)

# train our network
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

for i_test in range(5):
    state0 = env.reset()
    for i_round in range(200):
        env.render()
        action = dnq.act(state0)
        state1, _, done, _ = env.step(action)
        if done:
            print("testing episode {} finish after {} steps".format(i_test, i_round))
            break
        state0 = state1