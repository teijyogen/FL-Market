import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import time
from aggregation import error_bound_by_plosses_batch, aggr_batch, error_bound_by_plosses_weights_batch
from FL import *
from datasets import generate_dataset, Dataloader, load_bank, load_kdd99, generate_dataset_from_json
from torchvision import datasets, transforms
from regretnet import *
from utils import *
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DataParallel
import math
from client import Clients
from singleminded import baseline_batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
try:
     mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class Exp_Args():
    def __init__(self):
        self.n_agents = 10
        self.budget_rate_step = 0.1
        self.max_n_agents = 10
        self.n_items_ls = [5, 10, 15, 20]
        self.device_rank = 0
        self.dataset = "Banking"
        self.n_runs = 10
        self.batch = -1
        self.min_budget_rate = 0.1
        self.max_budget_rate = 2.0
        self.vary_budget = False
        self.rnd_step = 10
        self.n_gpus = 2
        self.n_processes = 4
        self.n_rounds = 100

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def map_data_dir(dataset_name, iid):
    if dataset_name == "Bank":
        if iid:
            return make_dir("data/bank/iid/")
        else:
            return make_dir("data/bank/niid/")
    elif dataset_name == "NSL-KDD":
        if iid:
            return make_dir("data/nslkdd/iid/")
        else:
            return make_dir("data/nslkdd/niid/")
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined")

def map_result_dir(dataset_name, iid, mech_name):
    if dataset_name == "Bank":
        if iid:
            return make_dir(f"run/bank/iid/{mech_name}/")
        else:
            return make_dir(f"run/bank/niid/{mech_name}/")
    elif dataset_name == "NSL-KDD":
        if iid:
            return make_dir(f"run/nslkdd/iid/{mech_name}/")
        else:
            return make_dir(f"run/nslkdd/niid/{mech_name}/")
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined")

def map_abbr_name(auc_mech_name, aggr_mech_name):
    if auc_mech_name == "RegretNet":
        abbr_auc_name = "reg"
    elif auc_mech_name == "M-RegretNet":
        abbr_auc_name = "m-reg"
    elif auc_mech_name == "DM-RegretNet":
        abbr_auc_name = "dm-reg"
    elif auc_mech_name == "All-in":
        abbr_auc_name = "allin"
    elif auc_mech_name == "FairQuery":
        abbr_auc_name = "fairq"
    else:
        raise ValueError(f"Auction {auc_mech_name} is not defined")

    if aggr_mech_name == "OptAggr":
        abbr_aggr_name = "opt"
    elif aggr_mech_name == "ConvlAggr":
        abbr_aggr_name = "convl"
    else:
        raise ValueError(f"Aggregation {aggr_mech_name} is not defined")

    return f"{abbr_auc_name}_{abbr_aggr_name}"

def map_labels(trade_mech_ls):
    labels = []
    for trade_mech in trade_mech_ls:
        label = ""
        if trade_mech[0] == "DM-RegretNet":
            label += r"$\bf{DM}$-$\bf{RegretNet}$"
        elif trade_mech[0] == "All-in":
            label += r"$\bf{All-in}$"
        else:
            label += trade_mech[0]

        if trade_mech[1] == "OptAggr":
            label += r"+$\bf{OptAggr}$"
        else:
            label += "+%s" %(trade_mech[1])

        labels.append(label)

    return labels


def load_auc_model(model_name):
    model_dict = torch.load(model_name)
    arch = model_dict["arch"]
    state_dict = model_dict["state_dict"]
    model = RegretNet(arch["n_agents"], arch["n_items"], activation=arch["activation"], hidden_layer_size=arch["hidden_layer_size"],
                      n_hidden_layers=arch["n_hidden_layers"], p_activation=arch["p_activation"],
                      a_activation=arch["a_activation"], separate=arch["separate"], normalized_input=arch["normalized_input"])
    model = DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.module
    model.deter_train = False
    model.eval()
    # print(state_dict)

    return model

def auction(reports, budget, trade_mech, model=None, expected=False):
    batch_size = reports.shape[0]
    n_agents = reports.shape[1]
    n_items = reports.shape[2] - 2
    budget = budget.view(-1, 1)
    device = reports.device
    sizes = reports[:, :, -1].view(-1, n_agents)

    if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
        # print(budget.shape)
        reports[:, :, 0] = reports[:, :, 0] * reports[:, :, 2]
        plosses, payments = baseline_batch(reports, budget, method=trade_mech[0])
    else:
        reports = reports.reshape(-1, n_agents, n_items + 2)
        allocs, payments = model((reports, budget))
        pbudgets = reports.view(-1, n_agents, n_items + 2)[:, :, -2]

        if expected:
            plosses = allocs_to_plosses(allocs, pbudgets)
        else:
            plosses, _ = allocs_instantiate_plosses(allocs, pbudgets)


    weights = aggr_batch(plosses, sizes, method=trade_mech[1])

    return plosses, weights


def acc_eval(plosses, weights, fl_model, local_sets, test_data, fl_args, multirnd=-1):
    acc_ls = []

    for rnd in range(fl_args.rounds):
        local_sets_rnd = local_sets[rnd]
        fl_model = ldp_fed_sgd(fl_model, fl_args, plosses[rnd, :], weights[rnd, :], local_sets_rnd, rnd)
        if multirnd:
            if rnd == 0 or rnd % multirnd == multirnd - 1:
                acc = test(fl_model, test_data, fl_args, rnd)
                acc_ls.append(acc)

    if not multirnd:
        acc_ls = [test(fl_model, test_data, fl_args, fl_args.rounds - 1)]

    return acc_ls


def acc_eval_mechs(trade_mech_ls, train_data, test_data, clients, fl_args, exp_args, run):
    torch.manual_seed((os.getpid() * int(time.time())) % 123456789)
    torch.cuda.manual_seed_all((os.getpid() * int(time.time())) % 123456789)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    local_sets = clients.return_local_sets_run(train_data, exp_args.n_agents, 0)
    del train_data
    fl_args.device_rank = run % exp_args.n_gpus
    torch.cuda.set_device(fl_args.device_rank)
    fl_model = Logistic(fl_args.input_size, fl_args.output_size).to(DEVICE)
    # fl_model = fl_model.to(DEVICE)
    if exp_args.vary_budget:
        acc_budget_mech_ls = []
        for trade_mech in trade_mech_ls:
            n_items = trade_mech[3]
            if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
                auc_model = None
            else:
                model_name = trade_mech[2]
                auc_model = load_auc_model(model_name).to(DEVICE)
            reports = clients.return_bids_run(n_items, 0)
            reports = torch.tensor(reports).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)[:, :, :-2]
            fl_args.device = reports.device
            acc_budget_ls = []
            for i in range(exp_args.nb_budget_rate):
                budget_rate = exp_args.budget_rate_step * (i + 1)
                model = copy.deepcopy(fl_model)
                max_cost = generate_max_cost(reports)
                budget = budget_rate * max_cost
                plosses, weights = auction(reports, budget, trade_mech, model=auc_model)
                acc = acc_eval(plosses, weights, model, local_sets, test_data, fl_args)[-1]
                acc_budget_ls.append(acc)
            acc_budget_mech_ls.append(acc_budget_ls)
        return acc_budget_mech_ls
    else:
        acc_mech_ls = []
        budget_rate = torch.rand((fl_args.rounds, 1)).to(DEVICE) * (exp_args.min_budget_rate - exp_args.max_budget_rate) + exp_args.max_budget_rate
        for trade_mech in trade_mech_ls:
            n_items = trade_mech[3]
            if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
                auc_model = None
            else:
                model_name = trade_mech[2]
                auc_model = load_auc_model(model_name).to(DEVICE)
            reports = clients.return_bids_run(n_items, 0)
            reports = torch.tensor(reports).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)[:, :, :-2]
            fl_args.device = reports.device
            model = copy.deepcopy(fl_model)
            max_cost = generate_max_cost(reports)
            budget = max_cost * budget_rate
            plosses, weights = auction(reports, budget, trade_mech, model=auc_model)
            accs = acc_eval(plosses, weights, model, local_sets, test_data, fl_args, multirnd=exp_args.rnd_step)
            acc_mech_ls.append(accs)

        return acc_mech_ls

def acc_eval_mechs_parallel(trade_mech_ls, title, file_name, labels, exp_args):

    fl_args = Arguments()
    fl_args.rounds = exp_args.n_rounds
    clients = Clients()
    clients.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    if exp_args.batch == -1:
        clients.filename = f"test_profiles_2mp.json"
    else:
        clients.filename = f"test_profiles_2mp_{exp_args.batch}.json"
    print(clients.filename)
    clients.load_json()
    if exp_args.dataset == "Bank":
        train_data, test_data = load_bank()
        fl_args.shape = (-1, 48)
        fl_args.input_size = 48
        fl_args.output_size = 2
    elif exp_args.dataset == "NSL-KDD":
        train_data, test_data = load_nslkdd()
        fl_args.shape = (-1, 122)
        fl_args.input_size = 122
        fl_args.output_size = 5
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined")

    nb_processes = exp_args.n_processes
    nb_pools = math.ceil(clients.n_runs / nb_processes)
    accs_pool_ls = []

    # fl_model = Logistic(fl_args.input_size, fl_args.output_size)

    for p in tqdm(range(nb_pools)):
        pool = mp.Pool(nb_processes)
        workers = []
        for run in range(p * nb_processes, min((p + 1) * nb_processes, clients.n_runs)):
            sub_clients = clients.return_clients_by_run(run)
            worker = pool.apply_async(acc_eval_mechs, args=(trade_mech_ls, train_data, test_data, sub_clients, fl_args, exp_args, run))
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            accs_pool_ls.append(worker.get())

    accs_np = np.array(accs_pool_ls)

    for i in range(len(trade_mech_ls)):
        trade_mech = trade_mech_ls[i]
        if len(trade_mech) > 4:
            content = accs_np[:, i, :].reshape(clients.n_runs, -1)
            dir = map_result_dir(exp_args.dataset, exp_args.iid, map_abbr_name(trade_mech[0], trade_mech[1]))
            np.save(dir+trade_mech[4], content)

    mechs_acc_ls = accs_np.mean(axis=0).tolist()
    if exp_args.vary_budget:
        budget_ls = [(b + 1.0) * exp_args.budget_rate_step for b in range(exp_args.nb_budget_rate)]
        mechs_budget_ls = [budget_ls for _ in trade_mech_ls]
        plot_budget_acc(mechs_budget_ls, mechs_acc_ls, labels, title, file_name, "linear")
    else:
        rnd_ls = [1] + [(r + 1) * exp_args.rnd_step for r in range(exp_args.n_rounds//exp_args.rnd_step)]
        mechs_rnd_ls = [rnd_ls for _ in trade_mech_ls]
        plot_rnd_acc(mechs_rnd_ls, mechs_acc_ls, labels, title, file_name, "linear")

def acc_load_npy(trade_mech_ls, title, file_name, labels, exp_args):

    mechs_acc_ls = []

    for trade_mech in trade_mech_ls:
        acc_ls = []
        dir = map_result_dir(exp_args.dataset, exp_args.iid, map_abbr_name(trade_mech[0], trade_mech[1]))

        for root, dirs, files in os.walk(dir):
            for f in files:
                print(f)
                data = np.load(os.path.join(root, f))
                print(data.shape)
                acc_ls.append(data)

        accs = np.concatenate(acc_ls, axis=0).mean(axis=0)
        mechs_acc_ls.append(accs)

    nb_x = len(mechs_acc_ls[0])
    if exp_args.vary_budget:
        budget_ls = np.arange(1, nb_x + 1) * exp_args.budget_rate_step
        mechs_budget_ls = [budget_ls for _ in mechs_acc_ls]
        plot_budget_acc(mechs_budget_ls, mechs_acc_ls, labels, title, file_name, "logit")
    else:
        rnd_ls = [1] + [(r + 1) * exp_args.rnd_step for r in range(exp_args.n_rounds//exp_args.rnd_step)]
        mechs_rnd_ls = [rnd_ls for _ in mechs_acc_ls]
        plot_rnd_acc(mechs_rnd_ls, mechs_acc_ls, labels, title, file_name, "linear")


def mse_eval(reports, budget, trade_mech, L=1.0, expected=False):
    if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
        auc_model = None
    else:
        auc_model = load_auc_model(trade_mech[2]).to(DEVICE)
    plosses, weights = auction(reports, budget, trade_mech, model=auc_model, expected=expected)
    sizes = reports[:, :, -1]
    error_bounds = error_bound_by_plosses_weights_batch(plosses, sizes, weights, L, train=False)
        # print(plosses[plosses.sum(dim=1) > 0.0])
        # print(weights[plosses.sum(dim=1) > 0.0])
        # print(error_bounds[plosses.sum(dim=1, keepdim=True) > 0.0])

    error_bounds[torch.isinf(error_bounds)] = -1
    error_bounds[torch.isnan(error_bounds)] = -1
    # print((error_bounds == -1).sum())

    return error_bounds[error_bounds>0.0]

def mse_budget(trade_mech_ls, title, file_name, labels, exp_args):
    clients = Clients()
    clients.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    clients.filename = f"test_profiles_2mp.json"
    proflies_nb = int(exp_args.n_runs * exp_args.n_rounds)
    clients.load_json()

    mechs_budget_ls = []
    mechs_error_bound_ls = []

    for trade_mech in trade_mech_ls:
        budget_ls = []
        error_bound_ls = []
        n_items = trade_mech[3]
        bid_data = torch.tensor(clients.return_bids(n_items)).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)
        bid_data = bid_data[:proflies_nb, :, :-2]
        data_loader = Dataloader(bid_data, 10000)

        for i in tqdm(range(exp_args.nb_budget_rate)):
            budget_rate = exp_args.budget_rate_step * (i + 1)
            error_bounds = np.array([])
            for j, reports in enumerate(data_loader):
                max_cost = generate_max_cost(reports)
                budget = budget_rate * max_cost
                error_bound = mse_eval(reports, budget, trade_mech)
                error_bound = error_bound.detach().to("cpu").numpy()
                error_bounds = np.append(error_bounds, error_bound)
            error_bound_mean = np.mean(error_bounds)

            error_bound_ls.append(error_bound_mean)
            budget_ls.append(budget_rate)

        mechs_budget_ls.append(budget_ls)
        mechs_error_bound_ls.append(error_bound_ls)

    plot_budget_mse(mechs_budget_ls, mechs_error_bound_ls, labels, title, file_name)

def mse_agents(trade_mech_ls, title, file_name, labels, exp_args):
    clients = Clients()
    clients.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    clients.filename = f"test_profiles_2mp.json"
    clients.load_json()
    proflies_nb = int(exp_args.n_runs * exp_args.n_rounds)

    mechs_n_agents_ls = []
    mechs_error_bound_ls = []

    nb_profiles = clients.n_runs * len(clients.data[0]) // exp_args.n_agents
    budget_rate = torch.rand((nb_profiles, 1)).to(DEVICE) * (
                exp_args.min_budget_rate - exp_args.max_budget_rate) + exp_args.max_budget_rate

    for trade_mech in trade_mech_ls:
        n_agents_ls = []
        error_bound_ls = []
        n_items = trade_mech[3]
        bid_data = torch.tensor(clients.return_bids(n_items)).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)
        bid_data = bid_data[:proflies_nb, :, :-2]
        batch_size = 10000
        data_loader = Dataloader(bid_data, batch_size)

        for i in tqdm(range(1, exp_args.max_n_agents)):
            n_agents = i + 1
            error_bounds = np.array([])
            for j, reports in enumerate(data_loader):
                reports[:, n_agents:, :] = reports[:, n_agents:, :] * 0.0
                max_cost = generate_max_cost(reports[:, :n_agents, :])
                budget = max_cost * budget_rate[j*batch_size:(j+1)*batch_size]
                if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
                    reports = reports[:, :n_agents, :]
                error_bound = mse_eval(reports, budget, trade_mech)
                error_bound = error_bound.detach().to("cpu").numpy()
                error_bounds = np.append(error_bounds, error_bound)
            error_bound_mean = np.mean(error_bounds)

            error_bound_ls.append(error_bound_mean)
            n_agents_ls.append(n_agents)

        mechs_n_agents_ls.append(n_agents_ls)
        mechs_error_bound_ls.append(error_bound_ls)

    plot_n_agents_mse(mechs_n_agents_ls, mechs_error_bound_ls, labels, title, file_name, "linear")

def guarantees_eval(reports, budget, val_type, trade_mech, misreport_iter=100, lr=1e-1):
    batch_size = reports.shape[0]
    misreports = reports.clone().detach().to(DEVICE)

    model_name = trade_mech[2]
    model = load_auc_model(model_name).to(DEVICE)
    optimize_misreports(model, reports, misreports, budget=budget, val_type=val_type, misreport_iter=misreport_iter, lr=lr, train=False, instantiation=True)
    allocs, payments = model((reports, budget))
    vals = reports[:, :, :-2]
    sizes = reports[:, :, -1]
    costs = torch.sum(allocs * vals, dim=2) * sizes

    truthful_util = calc_agent_util(reports, allocs, payments, instantiation=True)
    untruthful_util = tiled_misreport_util(misreports, reports, model, budget, val_type=val_type, instantiation=True)
    regrets = torch.clamp(untruthful_util - truthful_util, min=0)
    ir_violation = -torch.clamp(truthful_util, max=0)

    return regrets / costs, ir_violation / costs


def guarantees(trade_mech_ls, exp_args):
    regret_ls = []
    ir_ls = []
    m_ls = []

    clients = Clients()
    clients.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    clients.filename = f"test_profiles_2mp.json"
    clients.load_json()

    proflies_nb = int(exp_args.n_runs * exp_args.n_rounds)
    budget_rate = torch.rand((proflies_nb, 1)).to(DEVICE) * (
                exp_args.min_budget_rate - exp_args.max_budget_rate) + exp_args.max_budget_rate

    for trade_mech in tqdm(trade_mech_ls):

        n_items = trade_mech[3]
        m_ls.append(n_items)
        data = torch.tensor(clients.return_bids(n_items)).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)
        data = data[:proflies_nb, :, :]
        batch_size = min(10000, proflies_nb)
        data_loader = Dataloader(data, batch_size)

        if type(trade_mech[2]) != list:
            trade_mech[2] = [trade_mech[2]]
            print(trade_mech[2])

        regrets = np.array([])
        irs = np.array([])
        for model_name in trade_mech[2]:
            mech = (trade_mech[0], trade_mech[1], model_name)
            for j, batch in enumerate(data_loader):
                reports = batch[:, :, :-2]
                val_type = batch[:, :, -2:]
                max_cost = generate_max_cost(reports)
                budget = max_cost * budget_rate[j * batch_size:(j + 1) * batch_size]
                regret, ir = guarantees_eval(reports, budget, val_type, mech)
                regret = regret.detach().to("cpu").numpy()
                ir = ir.detach().to("cpu").numpy()
                regrets = np.append(regrets, regret)
                irs = np.append(irs, ir)

        regret_mean = regrets.mean()
        ir_mean = irs.mean()

        regret_ls.append(regret_mean)
        ir_ls.append(ir_mean)
    print("\n--------------------------------------------")
    print("\nNormalized regret vector")
    print(regret_ls)
    print("\nNormalized IR vio. vector")
    print(ir_ls)
    print("\n--------------------------------------------")
    return m_ls, regret_ls, ir_ls

def guarantees_plot(trade_mech_ls, title, file_name, exp_args):

    m_ls, regret_ls, ir_ls = guarantees(trade_mech_ls, exp_args)
    plot_m_guarantees([m_ls, m_ls], [regret_ls, ir_ls], ["regret", "IR violation"], title, file_name, yscale='linear')

def invalid_rate_budget(trade_mech_ls, title, file_name, labels, exp_args):
    clients = Clients()
    clients.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    clients.filename = f"test_profiles_2mp.json"
    clients.load_json()
    proflies_nb = int(exp_args.n_runs * exp_args.n_rounds)

    mechs_budget_ls = []
    mechs_invalid_rate_ls = []
    for trade_mech in trade_mech_ls:
        n_items = trade_mech[3]
        data = torch.tensor(clients.return_bids(n_items)).float().to(DEVICE).reshape(-1, exp_args.n_agents, n_items + 4)
        data = data[:proflies_nb, :, :-2]
        data_loader = Dataloader(data, 100000)

        model_name = trade_mech[2]
        model = load_auc_model(model_name).to(DEVICE)

        budget_ls = []
        invalid_rate_ls = []

        for i in tqdm(range(exp_args.nb_budget_rate)):
            budget_rate = exp_args.budget_rate_step * (i + 1)
            invalid_rates = np.array([])
            for j, reports in enumerate(data_loader):
                max_cost = generate_max_cost(reports)
                budget = budget_rate * max_cost
                allocs, payments = model((reports, budget))
                full_allocs = calc_full_allocs(allocs)
                invalid_rate = torch.prod(full_allocs[:, :, 0], dim=1)
                invalid_rate = invalid_rate.detach().to("cpu").numpy()
                invalid_rates = np.append(invalid_rates, invalid_rate)
            invalid_rate_mean = invalid_rates.mean()

            invalid_rate_ls.append(invalid_rate_mean)
            budget_ls.append(budget_rate)

        mechs_budget_ls.append(budget_ls)
        mechs_invalid_rate_ls.append(invalid_rate_ls)

    plot_budget_invalid_rate(mechs_budget_ls, mechs_invalid_rate_ls, labels, title, file_name)

if __name__ == '__main__':


    trade_mech_ls = [
        FQ_CONVL + ["nslkdd_iid.npy"],
        ALLIN_CONVL + ["nslkdd_iid.npy"],
        FQ_OPT + ["nslkdd_iid.npy"],
        ALLIN_OPT + ["nslkdd_iid.npy"]
    ]

    labels = map_labels(trade_mech_ls)
    exp_args = Exp_Args()
    exp_args.n_runs = 100
    exp_args.dataset = "NSL-KDD"
    exp_args.iid = True
    exp_args.budget_rate_step = 0.1
    exp_args.nb_budget_rate = 20
    exp_args.max_budget_rate = 2.0

    # mse_agents(trade_mech_ls, "NSL-KDD (IID)", "figure/n_err_single_nslkdd_iid.png", labels, exp_args)
    # acc_eval_mechs_parallel(trade_mech_ls, "NSL-KDD (IID)", "figure/acc_single_nslkdd_iid.png", labels, exp_args)
    # mse_budget(trade_mech_ls, "NSL-KDD (IID)", "figure/b_err_single_nslkdd_iid.png", labels, exp_args)
    # acc_load_npy(trade_mech_ls, "NSL-KDD (Non-IID)", "figure/acc_single_nslkdd_niid.png", labels, exp_args)


    # trade_mech_ls = [
    #     REG_CONVL_BANK_IID,
    #     TREG_CONVL_BANK_IID,
    #     FREG_CONVL_BANK_IID,
    #     MREG_CONVL_BANK_IID,
    #     STREG_CONVL_BANK_IID,
    # ]


    trade_mech_ls = [
        REG_CONVL_BANK_IID + ["bank_iid.npy"],
        MREG_CONVL_BANK_IID + ["bank_iid.npy"],
        DM_CONVL_BANK_IID + ["bank_iid.npy"],
        REG_OPT_BANK_IID + ["bank_iid.npy"],
        MREG_OPT_BANK_IID + ["bank_iid.npy"],
        DM_OPT_BANK_IID + ["bank_iid.npy"]
    ]

    labels = map_labels(trade_mech_ls)
    exp_args = Exp_Args()
    exp_args.n_runs = 100
    exp_args.dataset = "Bank"
    exp_args.iid = True
    exp_args.budget_rate_step = 0.1
    exp_args.nb_budget_rate = 20
    exp_args.max_budget_rate = 2.0


    # mse_budget(trade_mech_ls, "BANK (IID)", "figure/b_err_general_bank_iid_50r.png", labels, exp_args)
    # guarantees(trade_mech_ls, exp_args)
    # guarantees_plot(trade_mech_ls, "BANK (IID)", "figure/guarantee_bank_iid_2mp_2mb_mo_0625_50r.png", exp_args)
    # acc_eval_mechs_parallel(trade_mech_ls, "BANK (IID)", "figure/acc_general_bank_iid_50r.png", labels, exp_args)
    # acc_load_npy(trade_mech_ls, "NSL-KDD (Non-IID)", "figure/acc_general_bank_iid_50r.png", labels, exp_args)



    # trade_mech_ls = [
    #     REG_CONVL_NSLKDD_IID,
    #     TREG_CONVL_NSLKDD_IID,
    #     FREG_CONVL_NSLKDD_IID,
    #     MREG_CONVL_NSLKDD_IID,
    #     STREG_CONVL_NSLKDD_IID,
    # ]

    trade_mech_ls = [
        REG_CONVL_NSLKDD_IID + ["nslkdd_iid.npy"],
        MREG_CONVL_NSLKDD_IID + ["nslkdd_iid.npy"],
        DM_CONVL_NSLKDD_IID + ["nslkdd_iid.npy"],
        # REG_OPT_NSLKDD_IID + ["nslkdd_iid.npy"],
        # MREG_OPT_NSLKDD_IID + ["nslkdd_iid.npy"],
        DM_OPT_NSLKDD_IID + ["nslkdd_iid.npy"]
    ]

    lables_for_invalid_rate = [
        "RegretNet",
        "M-RegretNet",
        r"$\bf{DM}$-$\bf{RegretNet}$+ConvlAggr",
        r"$\bf{DM}$-$\bf{RegretNet}$+$\bf{OptAggr}$"
    ]

    labels = map_labels(trade_mech_ls)
    exp_args = Exp_Args()
    exp_args.n_runs = 1000
    exp_args.dataset = "NSL-KDD"
    exp_args.iid = True
    exp_args.budget_rate_step = 0.1
    exp_args.nb_budget_rate = 20
    exp_args.max_budget_rate = 2.0

    # mse_budget(trade_mech_ls, "NSL-KDD (IID)", "figure/b_err_general_nslkdd_iid_50r.png", labels, exp_args)
    # mse_agents(trade_mech_ls, "NSL-KDD (IID)", "figure/n_err_general_nslkdd_iid_50r.png", labels, exp_args)
    guarantees(trade_mech_ls, exp_args)
    # guarantees_plot(M_EFFECT_NSLKDD_IID, "NSL-KDD (IID)", "figure/guarantee_nslkdd_iid_50r.png", exp_args)
    # invalid_rate_budget(trade_mech_ls, "NSL-KDD (IID)", "figure/invalid_nslkdd_iid_50r.png", lables_for_invalid_rate, exp_args)
    # acc_eval_mechs_parallel(trade_mech_ls, "NSL-KDD (IID)", "figure/acc_general_nslkdd_iid_50r.png", labels, exp_args)
    # acc_load_npy(trade_mech_ls, "NSL-KDD (Non-IID)", "figure/acc_general_nslkdd_iid_50r.png", labels, exp_args)

















