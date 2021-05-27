import gym
import random
import torch
import torch.nn as nn
import math
from dqn import ENVDQN

# Acrobot is a 2-link pendulum with only the second joint actuated.
# Initially, both links point downwards. The goal is to swing the
# end-effector at a height at least the length of one link above the base.
# Both links can swing freely and can pass by each other, i.e., they don't
# collide when they have the same angle.
# **STATE:**
# The state consists of the sin() and cos() of the two rotational joint
# angles and the joint angular velocities :
# [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
# For the first link, an angle of 0 corresponds to the link pointing downwards.
# The angle of the second link is relative to the angle of the first link.
# An angle of 0 corresponds to having the same angle between the two links.
# A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
# **ACTIONS:**
# The action is either applying +1, 0 or -1 torque on the joint between
# the two pendulum links.

env = gym.make('Acrobot-v1')


cnt = 0
def finish_train(i_epi, i_step, done):
    global cnt
    if done:
        if i_step < 180:
            cnt += 1
        else:
            cnt = 0
        if cnt >= 3:
            return True
    return False

def reward_adjuster(reward, state0, state1, done):
    if done:
        reward += 1000
    # print(state0, reward)
    return reward + abs(state1[1])*5 + abs(state1[3])*5

envdqn = ENVDQN(env, 80, 1, reward_adjuster, None, finish_train)

envdqn.train()

envdqn.test()