import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tqdm.notebook import trange, tqdm

path_datasets = Path('./datasets')

ls = [f for f in path_datasets.glob('*') if f.is_file()]
_data = np.loadtxt(ls[1], skiprows=1, delimiter=',')

X = _data[:, 1:]
Y = _data[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_test, X_eval, Y_test, Y_eval = train_test_split(X_test, Y_test, test_size=0.5)

import torch
from torch import nn
import torch.nn.functional as F

# # check if torch supports mac gpu
# if torch.backends.mps.is_available() and\
#     torch.backends.mps.is_built():
#     device = torch.device("mps")
    
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
model = TinyModel()
model.to(device)

BATCH_SIZE = 32
optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
losses, accuracies = [], []
for i in range(100):
    model.train()
    samp = np.random.choice(range(X_train.shape[0]), size = BATCH_SIZE, replace=False)
    X = torch.tensor(X_train[samp].reshape((len(samp), 1, 28, 28)), dtype=torch.float32).to(device)
    Y = torch.tensor(Y_train[samp], dtype=torch.int).to(device)
    optim.zero_grad() #zero_grad
    out = model(X)
    loss = F.nll_loss(out, Y, reduction='sum')
    
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == Y).float().mean()
    
    loss.backward()
    optim.step()
    loss, accuracy = loss.item(), accuracy.item()
    losses.append(loss)
    accuracies.append(accuracy)

print(f"{losses=}, {accuracies=}")    

# plt.ylim(-0.1,1)
# plt.plot(losses)
# plt.plot(accuracies)
# plt.show()
