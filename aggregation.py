import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
# from qpth.qp import QPFunction, SpQPFunction
# from enum import Enum


def var_opt_aggr_batch(plosses):
    ad_plosses = torch.where(plosses.sum(dim=1, keepdim=True) > 0.0, plosses, plosses + 1.0)

    nume_weights = ad_plosses ** 2
    nume_weights[torch.isinf(nume_weights)] = 0.0
    nume_weights[torch.isnan(nume_weights)] = 0.0
    nume_weights[nume_weights.sum(dim=1) == 0.0] = 1.0

    deno_weight = nume_weights.sum(dim=1, keepdim=True)
    weights = nume_weights / deno_weight

    return weights

def data_size_aggr_batch(plosses, sizes):
    ad_plosses = torch.where(plosses.sum(dim=1, keepdim=True) > 0.0, plosses, plosses + 1.0)

    weights = torch.where(ad_plosses > 0.0, ad_plosses * 0.0 + sizes, ad_plosses * 0.0)
    weights = weights / weights.sum(dim=1, keepdim=True)

    return weights

def diffcp_aggr_batch(plosses_batch, sizes_batch, L=1.0):

    n_batch = plosses_batch.shape[0]
    ad_plosses_batch = torch.where(plosses_batch.sum(dim=1, keepdim=True) > 0.0, plosses_batch, plosses_batch + 1.0)
    weights_batch = torch.where(ad_plosses_batch > 0.0, ad_plosses_batch * 0.0 + 1.0, ad_plosses_batch * 0.0)


    for i in range(n_batch):
        sub_plosses = ad_plosses_batch[i][ad_plosses_batch[i] > 0.0]
        sub_sizes = sizes_batch[i][ad_plosses_batch[i] > 0.0]
        weights_batch[i][weights_batch[i] > 0.0] = diffcp_aggr(sub_plosses, sub_sizes, L=L)

    return weights_batch

def diffcp_aggr(plosses, sizes, L=1.0):
    sensi = 2 * L
    device = plosses.device

    vars = (2 * (sensi / plosses) ** 2)
    n_agents = vars.shape[0]

    Q1 = torch.diag_embed(vars.detach()).cpu()
    Q2 = torch.zeros(n_agents, n_agents).cpu()
    Q3 = torch.zeros(n_agents, n_agents).cpu()
    Q4 = (torch.ones(n_agents, n_agents) * (L ** 2)).cpu()
    Q12 = torch.cat((Q1, Q2), 1)
    Q34 = torch.cat((Q3, Q4), 1)
    Q = (torch.cat((Q12, Q34), 0)).reshape(2 * n_agents, 2 * n_agents)

    A1 = torch.ones(n_agents)
    A2 = torch.zeros(n_agents)
    A = torch.cat((A1, A2), 0).reshape(-1).cpu()

    G1 = torch.eye(n_agents)
    G2 = -torch.eye(n_agents)
    G3 = -torch.eye(n_agents)
    G4 = -torch.eye(n_agents)
    G12 = torch.cat((G1, G2), 1)
    G34 = torch.cat((G3, G4), 1)
    G = torch.cat((G12, G34), 0).cpu()

    w = sizes / sizes.sum()
    h1 = w
    h2 = -w
    h = torch.cat((h1, h2), 0).reshape(-1).cpu()

    h_cvxpy = cp.Parameter((2 * n_agents))
    x_cvxpy = cp.Variable((2 * n_agents))


    constrains = [A @ x_cvxpy == 1, G @ x_cvxpy <= h_cvxpy]
    objective = cp.Minimize(0.5 * cp.quad_form(x_cvxpy, Q))
    problem = cp.Problem(objective, constrains)

    try:
        cvxpylayer = CvxpyLayer(problem, parameters=[h_cvxpy], variables=[x_cvxpy])
    except ValueError:
        print("ValueError: Problem must be DPP.")
        return var_opt_aggr_batch(plosses.view(1, -1))

    try:
        solution, = cvxpylayer(h)
        weights = solution[:n_agents].to(device)
    except:
        print("SolverError")
        print(vars)
        if L == 0.00001:
            print("turn to VarOpt")
            return var_opt_aggr_batch(plosses.view(1, -1))
        weights = diffcp_aggr(plosses, sizes, L=L * 0.1)

    return weights.reshape(1, -1)

def error_bound_by_plosses_weights_batch(plosses_batch, sizes_batch, weights_batch, L=1.0, train=True):

    if torch.isinf(weights_batch).sum() > 0.0:
        print("weights contains inf")
        print(weights_batch)

    if torch.isnan(weights_batch).sum() > 0.0:
        print("weights contains nan")
        print(weights_batch)

    sensi = 2 * L

    if train:
        var_batch = ((weights_batch * sensi / plosses_batch) ** 2 * 2)
        var_batch[torch.isnan(var_batch)] = 0.0
        var_batch[torch.isinf(var_batch)] = 0.0
        var_batch = var_batch.sum(dim=1, keepdim=True)
        var_batch[torch.isnan(var_batch)] = 0.0
        var_batch[torch.isinf(var_batch)] = 0.0
        if torch.isinf(var_batch).sum() > 0.0:
            print("var contains inf")
            print(var_batch)

        if torch.isnan(var_batch).sum() > 0.0:
            print("var contains nan")
            print(var_batch)
    else:
        var_batch = ((weights_batch * sensi / plosses_batch) ** 2 * 2)
        var_batch[torch.isnan(var_batch)] = 0.0
        var_batch[torch.isinf(var_batch)] = 0.0
        var_batch = var_batch.sum(dim=1, keepdim=True)
        var_batch = torch.where(var_batch > 0.0, var_batch, 1 / var_batch)
        # var_batch = torch.where(var_batch > 0.0, var_batch, var_batch + (sensi / 0.01) ** 2 * 2)
    # print("Var")
    # print(var_batch)

    w_batch = sizes_batch / sizes_batch.sum(dim=1, keepdim=True)
    quad_bias_batch = torch.abs(weights_batch - w_batch).sum(dim=1, keepdim=True) * L
    # print("Bias")
    # print(quad_bias_batch ** 2)

    error_batch = quad_bias_batch ** 2 + var_batch

    # print("Error")
    # print(error_batch)
    return error_batch

def error_bound_by_plosses_batch(plosses_batch, sizes_batch, L=1.0, method="OptAggr", train=True):

    if method == "OptAggr":
        weights_batch = diffcp_aggr_batch(plosses_batch, sizes_batch)
    elif method == 'VarOpt':
        weights_batch = var_opt_aggr_batch(plosses_batch)
    elif method == 'ConvlAggr':
        weights_batch = data_size_aggr_batch(plosses_batch, sizes_batch)
    else:
        raise ValueError(f"{method} aggregation is not defined")

    # print(weights_batch)
    error_batch = error_bound_by_plosses_weights_batch(plosses_batch, sizes_batch, weights_batch, L=L, train=train)

    return error_batch

def error_bound_by_allocs_batch(allocs_batch, pbudgets_batch, sizes_batch, L, method='OptAggr', train=True):
    n_agents = allocs_batch.shape[1]
    n_items = allocs_batch.shape[2]
    device = allocs_batch.device

    deno = torch.arange(n_items, 0, step=-1, dtype=allocs_batch.dtype, device=device)
    items = pbudgets_batch.view(-1, n_agents, 1).repeat(1, 1, n_items)
    items = items / deno

    privacy_losses_batch = torch.sum(allocs_batch * items, dim=2)
    error_batch = error_bound_by_plosses_batch(privacy_losses_batch, sizes_batch, L, method=method, train=train)

    return error_batch

def aggr_batch(plosses_batch, sizes_batch, method="OptAggr"):
    n_batch = plosses_batch.shape[0]
    n_agents = plosses_batch.shape[1]

    if method == "VarOpt":
        weights_batch = var_opt_aggr_batch(plosses_batch)
    elif method == "OptAggr":
        weights_batch = diffcp_aggr_batch(plosses_batch, sizes_batch)
    elif method == "ConvlAggr":
        weights_batch = data_size_aggr_batch(plosses_batch, sizes_batch)
    else:
        raise ValueError(f"{method} aggregation is not defined")

    return weights_batch






