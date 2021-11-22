import h5py
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import torch.utils.data as Data
import copy
import matplotlib.pyplot as plt
from torch.distributions import Laplace
import numpy as np


class Arguments():
    def __init__(self):
        self.local_batch_size = 10
        self.test_batch_size = 1000
        self.rounds = 100
        self.lr = 0.01
        self.no_cuda = False
        self.seed = 0
        self.log_interval = 50
        self.save_model = False
        self.submit_grad = True
        self.L = 1.0
        self.sensi = 2.0 * self.L
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

class Logistic(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sig(out)
        return out


def ldp_fed_sgd(model, args, plosses, weights, local_sets, rnd):
    #     torch.cuda.empty_cache()
    updates = []
    keys = list(model.state_dict().keys())
    device = plosses.device
    total_num_samples = 0
    plosses = plosses.view(-1)
    n_agents = plosses.shape[0]
    weights = weights.view(-1)
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(plosses)

    global_model = copy.deepcopy(model).to(device)
    state = global_model.state_dict()

    # step 1: selection

    if (plosses == 0.0).all():
        return model

    for i in range(n_agents):
        # step 2: broadcasting

        if plosses[i] > 0.0:
            local_model = copy.deepcopy(model).to(device)

            # step 3: local training
            local_model.train()
            local_set = local_sets[i]

            X = local_set[0].to(device).float()
            Y = local_set[1].to(device).long()

            epsi = plosses[i]

            optimizer = optim.SGD(local_model.parameters(), lr=args.lr)
            optimizer.zero_grad()
            pred_Y = local_model(X)
            criter = nn.CrossEntropyLoss()
            loss = criter(pred_Y, Y)
            loss.backward()
            optimizer.step()

            # print('Round {}: Worker {} finished local training. \tLoss: {:.6f}'.format(
            #     rnd, i+1, loss.item()))

            # step 4: submission
            norm = nn.utils.clip_grad_norm_(local_model.parameters(), args.L, 1.0)

            for param in local_model.named_parameters():
                noised_grad = Laplace(param[1].grad, args.sensi / epsi).sample().to(device)
                state[param[0]] = state[param[0]] - noised_grad * weights[i] * args.lr


    global_model.load_state_dict(state)

    return global_model

def test(model, test_set, args, rnd):
    model.eval()
    device = args.device
    test_loss = 0.0
    correct = 0
    # print(len(test_set))
    data_loader = Data.DataLoader(dataset=test_set, batch_size=10000)
    num_samples = 0
    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            x = test_data[0].to(device).float()
            y = test_data[1].to(device).long()
            # x = test_data[0].to(device).view(-1, 784)
            # y = test_data[1].to(device)
            num_samples += len(x)
            pred_y = model(x)
            # test_loss += F.nll_loss(pred_y, y.long(), reduction='sum')
            criter = nn.CrossEntropyLoss()
            test_loss += criter(pred_y, y.long())
            pred = pred_y.argmax(1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= num_samples
        # print('\n Round: {}  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     rnd, test_loss, correct, num_samples,
        #     100. * correct / num_samples))
    return correct / num_samples

