import torch
import random
import math
from torch.utils.data import Dataset
import numpy as np
import json


class H5Dataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


def load_bank():
    data = np.loadtxt(open("data/bank-additional-full.csv", "rb"), delimiter=",", skiprows=1)
    np.random.shuffle(data)
    train_x = np.concatenate([data[0:30000, 0:14], data[0:30000, 15:]],axis=1)
    train_x = torch.tensor(train_x)
    train_y = data[0:30000, 14]
    train_y = torch.tensor(train_y)
    train_set = H5Dataset(train_x, train_y)

    test_x = np.concatenate([data[30000:40000, 0:14], data[30000:40000, 15:]],axis=1)
    test_x = torch.tensor(test_x)
    test_y = data[30000:40000, 14]
    test_y = torch.tensor(test_y)
    test_set = H5Dataset(test_x, test_y)

    return train_set, test_set


class Dataloader(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.size = data.size(0)
        self.data = data
        self.iter = 0

    def _sampler(self, size, batch_size, shuffle=True):
        if shuffle:
            idxs = torch.randperm(size)
        else:
            idxs = torch.arange(size)
        for batch_idxs in idxs.split(batch_size):
            yield batch_idxs

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == 0:
            self.sampler = self._sampler(self.size, self.batch_size, shuffle=self.shuffle)
        self.iter = (self.iter + 1) % (len(self)+1)
        idx = next(self.sampler)
        return self.data[idx]

    def __len__(self):
        return (self.size-1)//self.batch_size+1


def generate_dataset(n_agents, n_items, num_examples, max_pbudget=5.0, min_pbudget=0.1):
    example_dists = []
    for j in range(num_examples):
        agent_dists = []
        for i in range(n_agents):
            pbudget = random.uniform(min_pbudget, max_pbudget)
            cost_selection = random.choice(["grad", "sqrt", "expo", "linear"])
            if cost_selection == "grad":
                def c(x):
                    return x**2
            elif cost_selection == "sqrt":
                def c(x):
                    return 2*x**0.5
            elif cost_selection == "linear":
                def c(x):
                    return 2*x
            else:
                def c(x):
                    return math.exp(x)-1
            alph = random.uniform(0.5, 1.5)
            item_dists = []
            for k in range(n_items):
                item = (k + 1) * pbudget / n_items
                item_dists.append(alph * c(item))
            item_dists.append(pbudget)
            agent_dists.append(item_dists)
        example_dists.append(agent_dists)

    return torch.tensor(example_dists)


def generate_dataset_output_json(n_agents, num_examples, file_name, max_pbudget=5.0, min_pbudget=0.1):
    profiles_dict = {}
    for j in range(num_examples):
        profile_dict = {}
        for i in range(n_agents):
            profile_dict["client "+str(i)] = {
                "pbudget": random.uniform(min_pbudget, max_pbudget),
                "val func type": random.choice(["grad", "sqrt", "expo", "linear"]),
                "ratio": random.uniform(0.5, 1.5)
            }

        profiles_dict["profile "+str(j)] = profile_dict

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(profiles_dict, indent=4))


def generate_dataset_from_json(file_name, n_items):
    with open(file_name, 'r', encoding='utf8') as f:
        profiles_dict = json.load(f)

    example_dists = []
    for j in range(len(profiles_dict)):
        agent_dists = []
        profile_dict = profiles_dict["profile "+str(j)]

        for i in range(len(profile_dict)):
            agent_dict = profile_dict["client "+str(i)]
            pbudget = agent_dict["pbudget"]
            cost_selection = agent_dict["val func type"]
            ratio = agent_dict["ratio"]

            if cost_selection == "grad":
                def c(x):
                    return x ** 2
            elif cost_selection == "sqrt":
                def c(x):
                    return 2 * x ** 0.5
            elif cost_selection == "linear":
                def c(x):
                    return 2*x
            else:
                def c(x):
                    return math.exp(x) - 1
            item_dists = []
            for k in range(n_items):
                item = (k + 1) * pbudget / n_items
                item_dists.append(ratio * c(item))

            item_dists.append(pbudget)
            agent_dists.append(item_dists)
        example_dists.append(agent_dists)

    return torch.tensor(example_dists)

