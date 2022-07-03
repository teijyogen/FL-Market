import torch
import random
import os
import pickle
from operator import itemgetter
from torchvision import datasets, transforms
import numpy as np
import json
import math
from tqdm import tqdm
import multiprocessing as mp
from datasets import load_bank, load_kdd99, load_nslkdd
import torch.utils.data as Data
import time

def extr_noniid_dirt(dataset, n_clients, n_classes, alpha=0.5):
    data_size = len(dataset)
    idxs_client_dict = {i: [] for i in range(n_clients)}
    idxs = np.arange(data_size)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)
    labels = idxs_labels[1, :]
    data_size_each_class = np.array([np.sum(labels == c) for c in range(n_classes)]).reshape(n_classes, 1)

    # divide and assign
    idxs_classes = []
    for j in range(n_classes):
        idxs_classj = idxs[data_size_each_class[:j].sum() : data_size_each_class[:j+1].sum()].tolist()
        idxs_classes.append(idxs_classj)

    max_class = np.argmax(data_size_each_class)


    # for i in range(n_clients):
    #     rand_set = np.random.choice(idxs_classes[max_class], 10, replace=False).tolist()
    #     idxs_classes[max_class] = list(set(idxs_classes[max_class]) - set(rand_set))
    #     idxs_client_dict[i] = idxs_client_dict[i] + rand_set
    # data_size_each_class[max_class] -= 10 * n_clients

    distribution = np.random.dirichlet(np.repeat(alpha, n_clients), size=n_classes).astype(np.float64)
    data_size_each_class[max_class] -= 10 * n_clients
    data_size_each_class_client = (distribution * data_size_each_class).astype(int)
    data_size_each_class_client[max_class] += 10

    for i in range(n_clients):
        for j in range(n_classes):
            if i == n_clients - 1:
                rand_set = idxs_classes[j]
            else:
                rand_set = np.random.choice(idxs_classes[j], data_size_each_class_client[j, i], replace=False).tolist()
            idxs_classes[j] = list(set(idxs_classes[j]) - set(rand_set))
            idxs_client_dict[i] = idxs_client_dict[i] + rand_set


    return idxs_client_dict

class Client:
    def __init__(self, idx, dataset_name="", data_indices=None, data_size=None, val_func_type=None, privacy_budget=None, factor=None):
        self.idx = idx
        self.dataset_name = dataset_name
        self.data_indices = data_indices
        self.data_size = data_size
        self.val_func_type = val_func_type
        self.privacy_budget = privacy_budget
        self.factor = factor

    def __int__(self, **attrs):
        self.__dict__.update(attrs)


    def data(self, dataset):
        subset = torch.utils.data.Subset(dataset, self.data_indices)
        data_loader = Data.DataLoader(subset, batch_size=len(subset))
        for data in data_loader:
            return data

    def return_bid(self, n_items):
        if self.val_func_type == "grad":
            def v(x):
                return x ** 2
            type = 0
        elif self.val_func_type == "sqrt":
            def v(x):
                return 2 * x ** 0.5
            type = 1
        elif self.val_func_type == "linear":
            def v(x):
                return 2 * x
            type = 2
        else:
            def v(x):
                return math.exp(x) - 1
            type = 3

        bid = []
        for k in range(n_items):
            item = (k + 1) * self.privacy_budget / n_items
            bid.append(self.factor * v(item))

        bid.append(self.privacy_budget)
        bid.append(self.data_size)
        bid.append(type)
        bid.append(self.factor)

        return bid

    def return_dict(self):
        client_dict = {}
        client_dict.update(self.__dict__)
        return client_dict


class Clients:
    def __init__(self):
        self.data = []
        self.n_runs = 0
        self.dirs = "data/"
        self.filename = ""
        self.min_n_samples = 10
        self.min_pbudget = 1.0
        self.max_pbudget = 5.0
        self.min_factor = 0.5
        self.max_factor = 1.5

    def generate_clients(self, dataset_name, n_profiles, n_clients_per_profile, iid=True, alpha=0.5):
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        random.seed((os.getpid() * int(time.time())) % 123456789)
        if dataset_name == "Bank":
            train_data, _ = load_bank()
            n_classes = 2
        elif dataset_name == "KDD99":
            train_data, _ = load_kdd99()
            n_classes = 5
        elif dataset_name == "NSL-KDD":
            train_data, _ = load_nslkdd()
            n_classes = 5
        else:
            raise ValueError(f"Dataset {dataset_name} is not defined")

        n_clients = n_profiles * n_clients_per_profile
        data_size = len(train_data)

        if iid:
            idxs = np.arange(data_size).tolist()
            dist = np.random.lognormal(0, 2.0, n_clients).astype(np.float64)
            n_samples_dist = ((data_size - self.min_n_samples * n_clients) * dist / dist.sum()).astype(int)
            n_samples_dist += self.min_n_samples

            clients_dict = {}
            for i in range(n_clients):
                if i == n_clients - 1:
                    samples = idxs
                else:
                    samples = np.random.choice(idxs, n_samples_dist[i], replace=False).tolist()
                idxs = list(set(idxs) - set(samples))
                val_func_type = random.choice(["grad", "sqrt", "expo", "linear"])
                privacy_budget = random.uniform(self.min_pbudget, self.max_pbudget)
                factor = random.uniform(self.min_factor, self.max_factor)
                client = Client(i, dataset_name, samples, len(samples), val_func_type, privacy_budget, factor)
                client_dict = client.return_dict()
                clients_dict[client.idx] = client_dict
        else:
            indices_ls = extr_noniid_dirt(train_data, n_clients, n_classes, alpha)
            clients_dict = {}
            for i in range(n_clients):
                val_func_type = random.choice(["grad", "sqrt", "expo", "linear"])
                privacy_budget = random.uniform(self.min_pbudget, self.max_pbudget)
                factor = random.uniform(self.min_factor, self.max_factor)
                client = Client(i, dataset_name, indices_ls[i], len(indices_ls[i]), val_func_type, privacy_budget,
                                factor)
                client_dict = client.return_dict()
                clients_dict[client.idx] = client_dict
        return clients_dict

    def generate_clients_mulruns(self, dataset_name, n_profiles, n_clients_per_profile, n_runs, iid=True, alpha=0.5, overlap=False):
        self.data = []
        n_processes = 10
        run_processes = n_processes

        clients_dict = {}

        times = int(math.ceil(n_runs / n_processes))
        for j in tqdm(range(times)):
            if j == times - 1 and n_runs % n_processes != 0:
                run_processes = n_runs % n_processes
                print(run_processes)
            pool = mp.Pool(run_processes)
            workers = []
            for i in range(run_processes):
                worker = pool.apply_async(self.generate_clients, args=(dataset_name, n_profiles, n_clients_per_profile, iid, alpha))
                workers.append(worker)

            pool.close()
            pool.join()

            for i in range(run_processes):
                sub_clients_dict = workers[i].get()
                clients_dict["run %s" % (j * n_processes + i + 1)] = sub_clients_dict

        self.save_json(clients_dict, overlap=overlap)
        self.n_runs = n_runs

    def save_json(self, clients_dict, overlap=False):
        if not os.path.exists(self.dirs):
            os.makedirs(self.dirs)

        if not overlap and os.path.exists(self.dirs+self.filename):
            print("Clients data exists")
            return

        with open(self.dirs+self.filename, 'w', encoding='utf-8') as f:
            content = json.dumps(clients_dict, indent=2)
            f.write(content)

    def load_json(self):
        with open(self.dirs+self.filename, 'r', encoding='utf8') as f:
            clients_dict = json.load(f)

        self.n_runs = len(clients_dict)
        self.data = []
        for i in range(self.n_runs):
            sub_clients_dict = clients_dict["run %s" %(i + 1)]
            sub_clients = []
            for client_dict in sub_clients_dict.values():
                client = Client(**client_dict)
                sub_clients.append(client)

            self.data.append(sub_clients)

    def return_bids(self, n_items):
        clients_ls = []
        for i in range(self.n_runs):
            sub_clients_ls = []
            for client in self.data[i]:
                sub_clients_ls.append(client.return_bid(n_items))
            clients_ls.append(sub_clients_ls)

        return np.array(clients_ls)

    def return_bids_run(self, n_items, run):
        clients_ls = []
        for client in self.data[run]:
            clients_ls.append(client.return_bid(n_items))

        return np.array(clients_ls)

    def return_local_sets_run(self, dataset, n_agents, run):
        local_sets = []
        clients = self.data[run]
        rnds = len(clients) // n_agents
        for rnd in range(rnds):
            local_sets_rnd = []
            for i in range(n_agents):
                client = clients[rnd * n_agents + i]
                local_sets_rnd.append(client.data(dataset))

            local_sets.append(local_sets_rnd)

        return local_sets

    def return_clients_by_run(self, run):
        clients = Clients()
        clients.data = [self.data[run]]
        clients.n_runs = 1
        clients.dirs = self.dirs
        clients.min_pbudget = self.min_pbudget
        clients.max_pbudget = self.max_pbudget
        clients.min_factor = self.min_factor
        clients.max_factor = self.max_factor
        clients.min_n_samples = self.min_n_samples
        return clients

if __name__ == '__main__':
    clients = Clients()
    clients.dirs = "data/nslkdd/iid/"
    clients.min_pbudget = 0.5
    clients.max_pbudget = 2.0

    # clients.filename = "train_profiles_2.json"
    # clients.generate_clients_mulruns("NSL-KDD", 100, 10, 1024, iid=True, overlap=True)
    clients.filename = "test_profiles_2mp.json"
    clients.generate_clients_mulruns("NSL-KDD", 100, 10, 1000, iid=True, overlap=True)
    # for run in range(10, 100):
    #     clients.filename = f"test_profiles_100r_{run}.json"
    #     clients.generate_clients_mulruns("NSL-KDD", 100, 10, 100, iid=False, overlap=True)

    clients = Clients()
    clients.dirs = "data/bank/iid/"
    clients.min_pbudget = 0.5
    clients.max_pbudget = 2.0

    # clients.filename = "train_profiles_2.json"
    # clients.generate_clients_mulruns("NSL-KDD", 100, 10, 1024, iid=True, overlap=True)
    clients.filename = "test_profiles_2mp.json"
    clients.generate_clients_mulruns("Bank", 100, 10, 1000, iid=True, overlap=True)
    # for run in range(10, 100):
    #     clients.filename = f"test_profiles_100r_{run}.json"
    #     clients.generate_clients_mulruns("NSL-KDD", 100, 10, 100, iid=False, overlap=True)





