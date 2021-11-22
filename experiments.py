import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from singleminded import baseline_batch
from aggregation import error_bound_by_plosses_batch, var_opt_aggr, error_opt_aggr, aggr_batch
from FL import *
from datasets import generate_dataset, Dataloader, load_bank, generate_dataset_from_json
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)


class Exp_Args():
    def __init__(self):
        self.n_agents = 10
        self.n_items = 20
        self.n_profiles = 10000
        self.max_pbudget = 5.0
        self.min_pbudget = 0.5
        self.L = 1.0
        self.sensi = 2 * self.L
        self.budget_rate_step = 0.1
        self.max_n_agents = 10
        self.n_items_ls = [5, 10, 15, 20]


def auction(reports, budget, trade_mech, expected=False):
    batch_size = reports.shape[0]
    n_agents = reports.shape[1]
    n_items = reports.shape[2] - 1
    budget = budget.view(-1, 1)

    if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
        # print(budget.shape)
        plosses, payments = baseline_batch(reports, budget, method=trade_mech[0])
    else:
        model_name = trade_mech[2]
        model_dict = torch.load(model_name)
        args = model_dict["args"]
        state_dict = model_dict["state_dict"]
        model = RegretNet(args.n_agents, args.n_items, activation='relu', hidden_layer_size=args.hidden_layer_size,
                          n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                          a_activation=args.a_activation, separate=args.separate).to(DEVICE)
        model.train = False
        model = DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module

        reports = reports[:, :, -(args.n_items+1):].reshape(-1, args.n_agents * (args.n_items+1))
        allocs, payments = model((reports, budget))
        pbudgets = reports.view(-1, args.n_agents, args.n_items+1)[:, :, -1]

        if expected:
            plosses = allocs_to_plosses(allocs, pbudgets)
        else:
            plosses = allocs_instantiate_plosses(allocs, pbudgets)


    weights = aggr_batch(plosses, method=trade_mech[1])


    plosses.to(DEVICE)
    weights.to(DEVICE)

    return plosses, weights


def acc_eval(reports, budget, fl_model, trade_mech, train_set, test_set, fl_args):
    batch_size = reports.shape[0]
    n_agents = reports.shape[1]
    n_items = reports.shape[2] - 1
    fl_args.device = DEVICE

    train_loader = Data.DataLoader(dataset=train_set, batch_size=fl_args.local_batch_size)
    data_iter = iter(train_loader)

    plosses, weights = auction(reports, budget, trade_mech)


    acc_ls = np.array([])

    for i in range(fl_args.rounds):
        rnd = i + 1
        local_sets = []
        for j in range(n_agents):
            try:
                local_set = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                local_set = next(data_iter)
            local_sets.append(local_set)


        fl_model = ldp_fed_sgd(fl_model, fl_args, plosses[i, :], weights[i, :], local_sets, rnd)
        acc = test(fl_model, test_set, fl_args, rnd)

        acc_ls = np.append(acc_ls, acc)

    return acc_ls


def acc_budget(trade_mech_ls, title, file_name, labels, exp_args):

    train_data, test_data = load_bank()
    model = Logistic(48, 2)
    fl_args = Arguments()
    fl_args.local_batch_size = int(len(train_data) / exp_args.n_agents / fl_args.rounds)
    nb_budget_rate = int(1.0 / exp_args.budget_rate_step)

    bid_data = generate_dataset(exp_args.n_agents, exp_args.n_items, exp_args.n_profiles,
                                    max_pbudget=exp_args.max_pbudget, min_pbudget=exp_args.min_pbudget).to(DEVICE)
    data_loader = Dataloader(bid_data, fl_args.rounds)

    mechs_acc_ls = []
    mechs_budget_ls = []

    for trade_mech in trade_mech_ls:
        acc_ls = []
        budget_ls = []

        for b in tqdm(range(nb_budget_rate)):
            budget_rate = (b + 1.0) * exp_args.budget_rate_step
            budget_ls.append(budget_rate)

            acc = 0.0
            for i, reports in enumerate(data_loader):
                fl_model = copy.deepcopy(model).to(DEVICE)
                budget = budget_rate * reports[:, :, -2].sum(dim=1).view(-1, 1)
                acc += acc_eval(reports, budget, fl_model, trade_mech, train_data, test_data, fl_args)[-1]

            acc /= len(data_loader)
            acc_ls.append(acc)

        mechs_acc_ls.append(acc_ls)
        mechs_budget_ls.append(budget_ls)

    plot_budget_acc(mechs_budget_ls, mechs_acc_ls, labels, title, file_name)


def mse_eval(reports, budget, trade_mech, sensi=2.0, L=1.0, expected=False):
    if trade_mech[0] == "All-in" or trade_mech[0] == "FairQuery":
        plosses, payments = baseline_batch(reports, budget, method=trade_mech[0])
    else:
        plosses, weights = auction(reports, budget, trade_mech, expected=expected)

    error_bounds = error_bound_by_plosses_batch(plosses, sensi, L, method=trade_mech[1], eval_mode=True)
    error_bounds[torch.isinf(error_bounds)] = -1
    error_bounds[torch.isnan(error_bounds)] = -1

    return error_bounds[error_bounds>0.0].mean()


def mse_budget(trade_mech_ls, title, file_name, labels, exp_args):
    data = generate_dataset(exp_args.n_agents, exp_args.n_items, exp_args.n_profiles,
                                max_pbudget=exp_args.max_pbudget, min_pbudget=exp_args.min_pbudget).to(DEVICE)
    data_loader = Dataloader(data, 10000)
    nb_budget_rate = int(1.0 / exp_args.budget_rate_step)

    mechs_budget_ls = []
    mechs_error_bound_ls = []

    for trade_mech in trade_mech_ls:
        budget_ls = []
        error_bound_ls = []

        for i in tqdm(range(nb_budget_rate)):
            budget_rate = exp_args.budget_rate_step * (i + 1)
            error_bounds = np.array([])
            for j, reports in enumerate(data_loader):
                budget = budget_rate * reports[:, :, -2].sum(dim=1).view(-1, 1)
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
    data = generate_dataset(exp_args.n_agents, exp_args.n_items, exp_args.n_profiles,
                                max_pbudget=exp_args.max_pbudget, min_pbudget=exp_args.min_pbudget).to(DEVICE)
    data_loader = Dataloader(data, 1000)

    mechs_n_agents_ls = []
    mechs_error_bound_ls = []
    for trade_mech in trade_mech_ls:
        n_agents_ls = []
        error_bound_ls = []

        for i in tqdm(range(1, exp_args.max_n_agents)):
            n_agents = i + 1
            error_bounds = np.array([])
            for j, reports in enumerate(data_loader):
                reports[:, n_agents:, :] = reports[:, n_agents:, :] * 0.0
                budget = 1.0 * reports[:, :n_agents, -2].sum(dim=1).view(-1, 1)
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

    plot_n_agents_mse(mechs_n_agents_ls, mechs_error_bound_ls, labels, title, file_name)



def incr_mse_items(trade_mech_models_ls, title, file_name, exp_args):

    n_items_ls = exp_args.n_items_ls
    mechs_error_bound_ls = []
    mechs_label_ls = []

    for trade_mech_models in trade_mech_models_ls:
        error_bound_ls = []

        i = 0
        for n_items in tqdm(n_items_ls):

            data = generate_dataset_from_json("data/test_profiles.json", n_items).to(DEVICE)
            data_loader = Dataloader(data, 1000)
            trade_mech = (trade_mech_models[0], trade_mech_models[1], trade_mech_models[2][i])

            incr_errs = np.array([])
            for j, reports in enumerate(data_loader):
                budget = reports[:, :, -2].sum(dim=1).view(-1, 1)
                budget = budget * (torch.rand(budget.shape).to(DEVICE) * (0.1 - 1.0) + 1.0)
                error_bound = mse_eval(reports, budget, trade_mech).detach().to("cpu").numpy()
                expected_error_bound = mse_eval(reports, budget, trade_mech, expected=True).detach().to("cpu").numpy()
                incr_err = error_bound - expected_error_bound
                print(incr_err)
                incr_errs = np.append(incr_errs, incr_err)
            incr_err_mean = np.mean(incr_errs)

            error_bound_ls.append(incr_err_mean)
            i += 1


        mechs_error_bound_ls.append(error_bound_ls)
        mechs_label_ls.append(trade_mech_models[0] + "+" + trade_mech_models[1])

    plot_bar(n_items_ls, mechs_error_bound_ls, mechs_label_ls, title, file_name, xlabel="M-unit", ylabel="introduced error")


def guarantees_eval(reports, budget, trade_mech, misreport_iter=100, lr=1e-1):
    batch_size = reports.shape[0]
    misreports = reports.clone().detach().to(DEVICE)

    model_name = trade_mech[2]
    model_dict = torch.load(model_name)
    args = model_dict["args"]
    state_dict = model_dict["state_dict"]
    model = RegretNet(args.n_agents, args.n_items, activation='relu', hidden_layer_size=args.hidden_layer_size,
                  n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                  a_activation=args.a_activation, separate=args.separate).to(DEVICE)
    model.train = False
    model = DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.module
    create_misreports(model, reports, misreports, budget=budget, misreport_iter=misreport_iter, lr=lr)
    allocs, payments = model((reports, budget))
    valuations = reports[:, :, :-1]
    costs = torch.sum(allocs * valuations, dim=2)

    truthful_util = calc_agent_util(reports, allocs, payments)
    untruthful_util = tiled_misreport_util(misreports, reports, model, budget)
    regrets = torch.clamp(untruthful_util - truthful_util, min=0)
    ir_violation = -torch.clamp(truthful_util, max=0)

    return regrets.mean() / costs.mean(), ir_violation.mean() / costs.mean()


def guarantees(trade_mech_ls, exp_args):
    regret_ls = []
    ir_ls = []

    for trade_mech in trade_mech_ls:

        n_items = trade_mech[3]

        data = generate_dataset(exp_args.n_agents, n_items, exp_args.n_profiles,
                                    max_pbudget=exp_args.max_pbudget, min_pbudget=exp_args.min_pbudget).to(DEVICE)
        data_loader = Dataloader(data, 100000)
        trade_mech = (trade_mech[0], trade_mech[1], trade_mech[2])

        regrets = np.array([])
        irs = np.array([])
        for j, reports in enumerate(data_loader):
            budget = reports[:, :, -2].sum(dim=1).view(-1, 1)
            budget = budget * (torch.rand(budget.shape).to(DEVICE) * (0.1 - 1.0) + 1.0)
            regret, ir = guarantees_eval(reports, budget, trade_mech)
            regret = regret.detach().to("cpu").numpy()
            ir = ir.detach().to("cpu").numpy()
            regrets = np.append(regrets, regret)
            irs = np.append(irs, ir)
        regret_mean = np.mean(regrets)
        ir_mean = np.mean(irs)

        regret_ls.append(regret_mean)
        ir_ls.append(ir_mean)

    print("\n--------------------------------------------")
    print("\nNormalized regret vector")
    print(regret_ls)
    print("\nNormalized IR vio. vector")
    print(ir_ls)
    print("\n--------------------------------------------")


def invalid_rate_eval(reports, budget, mechs):
    model_name = mechs[2]
    model_dict = torch.load(model_name)
    args = model_dict["args"]
    state_dict = model_dict["state_dict"]
    model = RegretNet(args.n_agents, args.n_items, activation='relu', hidden_layer_size=args.hidden_layer_size,
                  n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                  a_activation=args.a_activation, separate=args.separate).to(DEVICE)
    model.train = False
    model = DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.module
    reports = reports[:, :, -(args.n_items + 1):].reshape(-1, args.n_agents * (args.n_items + 1))
    allocs, payments = model((reports, budget))

    full_allocs = calc_full_allocs(allocs)
    invalid_rate = torch.prod(full_allocs[:, :, 0], dim=1).mean()

    return invalid_rate


def invalid_rate_budget(trade_mech_ls, title, file_name, labels, exp_args):
    data = generate_dataset(exp_args.n_agents, exp_args.n_items, exp_args.n_profiles,
                                max_pbudget=exp_args.max_pbudget, min_pbudget=exp_args.min_pbudget).to(DEVICE)
    data_loader = Dataloader(data, 1000)
    nb_budget_rate = int(1.0 / exp_args.budget_rate_step)

    mechs_budget_ls = []
    mechs_invalid_rate_ls = []
    for trade_mech in trade_mech_ls:
        budget_ls = []
        invalid_rate_ls = []

        for i in tqdm(range(nb_budget_rate)):
            budget_rate = exp_args.budget_rate_step * (i + 1)
            invalid_rates = np.array([])
            for j, reports in enumerate(data_loader):
                budget = budget_rate * reports[:, :, -2].sum(dim=1).view(-1, 1)
                invalid_rate = invalid_rate_eval(reports, budget, trade_mech)
                invalid_rate = invalid_rate.detach().to("cpu").numpy()
                invalid_rates = np.append(invalid_rates, invalid_rate)
            invalid_rate_mean = np.mean(invalid_rates)

            invalid_rate_ls.append(invalid_rate_mean)
            budget_ls.append(budget_rate)

        mechs_budget_ls.append(budget_ls)
        mechs_invalid_rate_ls.append(invalid_rate_ls)

    plot_budget_invalid_rate(mechs_budget_ls, mechs_invalid_rate_ls, labels, title, file_name)


'''
Experiment 1:
    xlabel: budget_rate
    ylable: accuracy
    legend: trading mechanisms
'''

## Figure 11a
# exp_args = Exp_Args()
# exp_args.n_profiles = 100000
# exp_args.max_pbudget = 1.0
# exp_args.min_pbudget = 0.1
# acc_budget([
#     ("FairQuery", "ConvlAggr"),
#     ("All-in", "ConvlAggr"),
#     ("FairQuery", "OptAggr"),
#     ("All-in", "OptAggr"),
# ], "Single Minded", "figure/budget_acc_single.png",
#     [
# r"FairQuery+ConvlAggr",
# r"$\bf{All}$-$\bf{in}$+ConvlAggr",
# r"FairQuery+$\bf{OptAggr}$",
# r"$\bf{All}$-$\bf{in}$+$\bf{OptAggr}$",
# ], exp_args)


# Figure 11b
# exp_args = Exp_Args()
# exp_args.n_profiles = 100000
# exp_args.max_pbudget = 5.0
# exp_args.min_pbudget = 0.5
#
# acc_budget([
#     ("RegretNet", "ConvlAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "ConvlAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "ConvlAggr", "model/dm-regretnet_convl.pt"),
#     ("RegretNet", "OptAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "OptAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "OptAggr", "model/dm-regretnet_opt.pt"),
# ], "General", "figure/budget_acc_general.png",
#         [
#     r"RegretNet+ConvlAggr",
#     r"M-RegretNet+ConvlAggr",
#     r"$\bf{DM}$-$\bf{RegretNet}$+ConvlAggr",
#     r"RegretNet+$\bf{OptAggr}$",
#     r"M-RegretNet+$\bf{OptAggr}$",
#     r"$\bf{DM}$-$\bf{RegretNet}$+$\bf{OptAggr}$",
#     ], exp_args)


'''
Experiment 2:
    xlabel: budget_rate
    ylable: finite error bound
    legend: trading mechanisms
'''

## figure 7a
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.max_pbudget = 1.0
# exp_args.min_pbudget = 0.1
# mse_budget([
#     ("FairQuery", "ConvlAggr"),
#     ("All-in", "ConvlAggr"),
#     ("FairQuery", "OptAggr"),
#     ("All-in", "OptAggr"),
# ], "Single Minded", "figure/budget_mse_single.png",
#     [
# r"FairQuery+ConvlAggr",
# r"$\bf{All}$-$\bf{in}$+ConvlAggr",
# r"FairQuery+$\bf{OptAggr}$",
# r"$\bf{All}$-$\bf{in}$+$\bf{OptAggr}$",
# ], exp_args)


## figure 7b
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.max_pbudget = 5.0
# exp_args.min_pbudget = 0.5
#
# mse_budget([
#     ("RegretNet", "ConvlAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "ConvlAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "ConvlAggr", "model/dm-regretnet_convl.pt"),
#     ("RegretNet", "OptAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "OptAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "OptAggr", "model/dm-regretnet_opt.pt"),
# ], "General", "figure/budget_err_general.png",
#         [
#     r"RegretNet+ConvlAggr",
#     r"M-RegretNet+ConvlAggr",
#     r"$\bf{DM}$-$\bf{RegretNet}$+ConvlAggr",
#     r"RegretNet+$\bf{OptAggr}$",
#     r"M-RegretNet+$\bf{OptAggr}$",
#     r"$\bf{DM}$-$\bf{RegretNet}$+$\bf{OptAggr}$",
#     ], exp_args)


'''
Experiment 3:
    xlabel: budget_rate
    ylable: percentage of invalid global gradients
    legend: trading mechanisms
'''

## figure 9
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.m_pbudget = 5.0
# exp_args.min_pbudget = 0.5
# invalid_rate_budget([
#         ("RegretNet", "OptAggr", "model/regretnet.pt"),
#         ("M-RegretNet", "OptAggr", "model/m-regretnet.pt"),
#         ("DM-RegretNet", "OptAggr", "model/dm-regretnet_opt.pt"),
# ], "General", "figure/budget_invalid_rate_general.png",
#             [
#         r"RegretNet",
#         r"M-RegretNet",
#         r"$\bf{DM}$-$\bf{RegretNet}$"
#         ], exp_args)


'''
Experiment 4:
    xlabel: n_agents
    ylable: finite error bound
    legend: trading mechanisms
'''

## figure 8a
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.max_pbudget = 1.0
# exp_args.min_pbudget = 0.1
# mse_agents([
#     ("FairQuery", "ConvlAggr"),
#     ("All-in", "ConvlAggr"),
#     ("FairQuery", "OptAggr"),
#     ("All-in", "OptAggr"),
# ], "Single Minded", "figure/n_agents_mse_single.png",
#     [
# r"FairQuery+ConvlAggr",
# r"$\bf{All}$-$\bf{in}$+ConvlAggr",
# r"FairQuery+$\bf{OptAggr}$",
# r"$\bf{All}$-$\bf{in}$+$\bf{OptAggr}$",
# ], exp_args)

## figure 8b
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.max_pbudget = 5.0
# exp_args.min_pbudget = 0.5
#
# mse_agents([
#     ("RegretNet", "ConvlAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "ConvlAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "ConvlAggr", "model/dm-regretnet_convl.pt"),
#     ("RegretNet", "OptAggr", "model/regretnet.pt"),
#     ("M-RegretNet", "OptAggr", "model/m-regretnet.pt"),
#     ("DM-RegretNet", "OptAggr", "model/dm-regretnet_opt.pt"),
# ], "General", "figure/n_err_general.png",
#         [
#             r"RegretNet+ConvlAggr",
#             r"M-RegretNet+ConvlAggr",
#             r"$\bf{DM}$-$\bf{RegretNet}$+ConvlAggr",
#             r"RegretNet+$\bf{OptAggr}$",
#             r"M-RegretNet+$\bf{OptAggr}$",
#             r"$\bf{DM}$-$\bf{RegretNet}$+$\bf{OptAggr}$",
#     ], exp_args)


'''
Experiment 4:
    xlabel: n_items
    ylable: introduced error
    legend: trading mechanisms
'''

## Figure 10
# exp_args = Exp_Args()
# exp_args.n_profiles = 10000
# exp_args.m_pbudget = 5.0
# exp_args.min_pbudget = 0.5
# incr_mse_items([
#         ("M-RegretNet", "OptAggr", [
# "model/5-regretnet.pt",
# "model/10-regretnet.pt",
# "model/15-regretnet.pt",
# "model/m-regretnet.pt",
#         ]),
#         ("M-RegretNet", "ConvlAggr", [
# "model/5-regretnet.pt",
# "model/10-regretnet.pt",
# "model/15-regretnet.pt",
# "model/m-regretnet.pt",
#         ]),
# ], "", "figure/n_items_incr_mse.png", exp_args)


'''
Experiment 5: Normalized regret and IR
'''
# exp_args = Exp_Args()
# exp_args.n_profiles = 100000
# exp_args.min_pbudget = 0.5
#
# guarantees([
#     ("RegretNet", "OptAggr", "model/regretnet.pt", 1),
#     ("RegretNet", "ConvlAggr", "model/regretnet.pt", 1),
#     ("M-RegretNet", "OptAggr", "model/m-regretnet.pt", 20),
#     ("M-RegretNet", "ConvlAggr", "model/m-regretnet.pt", 20),
#     ("D-RegretNet", "OptAggr", "model/dm-regretnet_opt.pt", 20),
#     ("D-RegretNet", "ConvlAggr", "model/dm-regretnet_convl.pt", 20),
# ], exp_args)



























