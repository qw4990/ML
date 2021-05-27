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
state_dim = 4
action_dim = 3


# train our network
for i_episode in range(1):
    state0 = env.reset()
    for t in range(100):
        env.render()
        action = random.randint(0, action_dim-1)
        if (t//20) % 2 == 0:
            action = 0
        else:
            action = 2
        print("===>>> ", action, (t/20), (t/20)%2)
        env.step(action)

