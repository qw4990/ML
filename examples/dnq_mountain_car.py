import gym
import random
import torch
import torch.nn as nn
from dqn import ENVDQN
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

init_pos = -0.5
def state_init(state):
    global init_pos
    init_pos = state[0]

def reward_adjuster(reward, state0, state1, done):
    if reward == 0:
        print("==>>> reach the flag")
        return 1000
    pos = state1[0]
    vel = state1[1]
    pos_diff = abs(init_pos - pos)
    return reward + pos_diff + abs(vel*5)

envdqn = ENVDQN(env, 100, 1, reward_adjuster)

envdqn.train()

envdqn.test()