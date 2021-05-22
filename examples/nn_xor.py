import torch
import torch.nn as nn
import numpy as np

# input dataset
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).float()
y = torch.tensor([[1.], [0.], [0.], [1.]]).float()
# print(x, y)

# x = np.mat('0 0;'
#            '0 1;'
#            '1 0;'
#            '1 1')
# x = torch.tensor(x).float()
# y = np.mat('1;'
#            '0;'
#            '0;'
#            '1')
# y = torch.tensor(y).float()
# print(x, y)


# construct the network
n = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
# print(n)

# set the optimizer
optimizer = torch.optim.SGD(n.parameters(), lr=0.05)
loss_func = nn.MSELoss()

# train the network
for epoch in range(5000):
    out = n(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
#     print(loss)
    optimizer.step()

# test it
print(n(x).data)