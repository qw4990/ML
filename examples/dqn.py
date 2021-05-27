import gym
import random
import torch
import torch.nn as nn
import math

class DQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.MLP = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.opt = torch.optim.Adam(self.MLP.parameters(), lr=0.001)
        self.steps = 0
        self.decay = 200
        self.eps_high = 0.9
        self.eps_low = 0.05
        self.GAMMA = 0.8
        self.tran_buffer = []
        self.BATCH_SIZE = 64
        self.BUFFER_CAP = 10000
        pass

    def act(self, state0):
        self.steps += 1
        epsilon = self.eps_low + (self.eps_high-self.eps_low) * (math.exp(-1.0 * self.steps/self.decay))
        act_benefits = self.MLP(torch.tensor(state0).float())
        if random.random() < epsilon: # to explore new states
            action = random.randint(0, self.action_dim-1)
        else: # use the action with the max benefit
            action = torch.argmax(act_benefits).item()
        return action

    def learn(self, state0, act, state1, reward):
        self.tran_buffer.append((state0, act, state1, reward))
        if len(self.tran_buffer) > self.BUFFER_CAP:
            self.tran_buffer.pop(0)
        if len(self.tran_buffer) < self.BATCH_SIZE:
            return

        samples = random.sample(self.tran_buffer, self.BATCH_SIZE)
        s0 = torch.tensor([s[0] for s in samples]).float()
        a0 = torch.tensor([s[1] for s in samples]).long().view(self.BATCH_SIZE, -1)
        s1 = torch.tensor([s[2] for s in samples]).float()
        r1 = torch.tensor([s[3] for s in samples]).float().view(self.BATCH_SIZE, -1)

        actual = r1 + self.GAMMA * torch.max(self.MLP(s1).detach(), dim=1)[0].view(self.BATCH_SIZE, -1)
        pred = self.MLP(s0).gather(1, a0)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, actual)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class ENVDQN:
    def __init__(self, env, reward_adjuster, train_epi, test_epi):
        self.dqn = DQN(env.observation_space.shape[0], env.action_space.n)
        self.env = env
        self.reward_adjuster = reward_adjuster
        self.train_epi = train_epi
        self.test_epi = test_epi

    def train(self):
        for i_epi in range(self.train_epi):
            state0 = self.env.reset()
            for i_round in range(200):
                action = self.dqn.act(state0)
                state1, reward, done, info = self.env.step(action)
                reward = self.reward_adjuster(reward, state0, state1, done)
                self.dqn.learn(state0, action, state1, reward)
                if done:
                    print("training episode {} finish after {} steps".format(i_epi, i_round))
                    break
                state0 = state1
    
    def test(self):
        for i_test in range(self.test_epi):
            state0 = self.env.reset()
            for i_round in range(200):
                self.env.render()
                action = self.dqn.act(state0)
                state1, _, done, _ = self.env.step(action)
                if done:
                    print("testing episode {} finish after {} steps".format(i_test, i_round))
                    break
                state0 = state1