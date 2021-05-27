"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.
State consists of STATE_W x STATE_H pixels.
The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.
The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.
Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.
To play yourself (it's rather fast for humans), type:
python gym/envs/box2d/car_racing.py
Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import gym
import random
import torch
import torch.nn as nn
import math
from dqn import ENVDQN

env = gym.make('CarRacing-v0')

state = env.reset()
for i in range(200):
    env.render()
    print(env.action_space)
    