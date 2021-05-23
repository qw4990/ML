# model: house_price = w_area*area + w_age*age + b
true_w = [2, -3.4]
true_b = 4.2

# generate input dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
xs = torch.randn(1000, 2).float()
ys = true_w[0]*xs[:, 0] + true_w[1]*xs[:, 1] + true_b
ys += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),dtype=torch.float32)
# plt.scatter(xs[:, 1].numpy(), ys.numpy(), 1);

# init parapeters
w = torch.tensor([1.0, 1.0], requires_grad=True)
b = torch.tensor(0., requires_grad=True)

# start to train
def model_cal(x, w, b):
    return x[0]*w[0]+x[1]*w[1]+b

for i in range(1000):
    x = xs[i]
    y = ys[i]
    est_y = model_cal(x, w, b)
    loss_val = (est_y-y)**2/2
    loss_val.backward()
    # adjust parameters by their grad
    w.data -= 0.03 * w.grad
    b.data -= 0.03 * b.grad
    w.grad.data.zero_() # reset the grad
    b.grad.data.zero_()
    
print(w.data, b.data)
print(true_w, true_b)