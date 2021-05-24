import gym
import random
import torch
import torch.nn as nn

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

# the network: 
#   input: the state
#   output: benefit of each action
#   model: linear + relu + linear
nw = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, action_dim))
opt = torch.optim.Adam(nw.parameters(), lr=0.001)
eps_high = 0.9
eps_low = 0.05
decay = 200
steps = 0
GAMMA = 0.8

buffer = []
BATCH_SIZE = 64
CAP = 10000

# train our network
for i_episode in range(80):
    state0 = env.reset()
    for t in range(200):
        # env.render()
        # pick an action
        steps += 1
        epsilon = eps_low + (eps_high-eps_low) * (math.exp(-1.0 * steps/decay))
        act_benefits = nw(torch.tensor(state0).float())
        if random.random() < epsilon: # to explore new states
            action = random.randint(0, action_dim-1)
        else: # use the action with the max benefit
            action = torch.argmax(act_benefits).item()
        state1, reward, done, info = env.step(action) # apply this action
        if done:
            reward = -10

        buffer.append((state0, action, state1, reward))
        if len(buffer) > CAP:
            buffer.pop(0)
        if len(buffer) >= BATCH_SIZE:
            samples = random.sample(buffer, BATCH_SIZE)
            s0 = torch.tensor([s[0] for s in samples]).float()
            a0 = torch.tensor([s[1] for s in samples]).long().view(BATCH_SIZE, -1)
            s1 = torch.tensor([s[2] for s in samples]).float()
            r1 = torch.tensor([s[3] for s in samples]).float().view(BATCH_SIZE, -1)

            actual = r1 + GAMMA * torch.max(nw(s1).detach(), dim=1)[0].view(BATCH_SIZE, -1)
            pred = nw(s0).gather(1, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, actual)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            break

        # move to the next state
        state0 = state1

