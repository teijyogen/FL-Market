import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def var_opt_aggr(plosses):
    if (plosses > 0.0).sum() == 0.0:
        weights = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
        weights[weights == 0.0] = 1 / weights.shape[0]
        return weights

    nz_plosses = plosses[plosses>0.]
    if torch.isinf(nz_plosses).sum() > 0.0:
        print("nz_plosses contains nan")
        print(nz_plosses)

    if torch.isnan(nz_plosses).sum() > 0.0:
        print("nz_plosses contains inf")
        print(nz_plosses)

    nume_weights = nz_plosses ** 2

    if torch.isinf(nume_weights).sum() > 0.0:
        print("nume_weights contains nan")
        print(nume_weights)

    if torch.isnan(nume_weights).sum() > 0.0:
        print("nume_weights contains inf")
        print(nume_weights)

    deno_weight = nume_weights.sum()
    nz_weights = nume_weights / deno_weight

    weights = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
    weights[weights > 0.0] = nz_weights

    return weights

def data_size_aggr(plosses):
    if (plosses > 0.0).sum() == 0.0:
        weights = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
        weights[weights == 0.0] = 1 / weights.shape[0]

        return weights

    weights = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
    weights[weights > 0.0] = 1 / (weights > 0.0).sum().item()

    return weights


def error_opt_aggr(plosses, L=1.0):
    if (plosses > 0.0).sum() == 0.0:
        weights = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
        weights[weights == 0.0] = 1 / weights.shape[0]
        # print("all-zero weights")
        return weights

    sensi = 2 * L
    device = plosses.device

    vars = torch.where(plosses > 0.0, plosses * 0.0 + 1.0, plosses * 0.0)
    vars[vars > 0.0] = (2 * (sensi / plosses[plosses>0.0]) ** 2)
    weights = torch.where(vars > 0.0, vars * 0.0 + 1.0, vars * 0.0)

    vars = vars[vars > 0.0]
    vars = vars.cpu()
    n_agents = vars.shape[0]

    x = cp.Variable(2 * n_agents)

    Q1 = torch.diag(vars).cpu()
    Q2 = torch.zeros(n_agents, n_agents).cpu()
    Q3 = torch.zeros(n_agents, n_agents).cpu()
    Q4 = (torch.ones(n_agents, n_agents) * (L ** 2)).cpu()
    Q12 = torch.cat((Q1, Q2), 1).cpu()
    Q34 = torch.cat((Q3, Q4), 1).cpu()
    Q = (torch.cat((Q12, Q34), 0)).view(2 * n_agents, 2 * n_agents)
    # Q.requires_grad = True

    Q_sqrt_cvxpy = cp.Parameter((2 * n_agents, 2 * n_agents))

    A1 = torch.ones(n_agents).cpu()
    A2 = torch.zeros(n_agents).cpu()
    A = torch.cat((A1, A2), 0).view(1, 2 * n_agents)

    b = torch.ones(1)

    G1 = torch.eye(n_agents).cpu()
    G2 = -torch.eye(n_agents).cpu()
    G3 = -torch.eye(n_agents).cpu()
    G4 = -torch.eye(n_agents).cpu()
    G12 = torch.cat((G1, G2), 1)
    G34 = torch.cat((G3, G4), 1)
    G = torch.cat((G12, G34), 0).view(2 * n_agents, 2 * n_agents)

    h1 = (torch.ones(n_agents) * (1 / n_agents)).cpu()
    h2 = (torch.ones(n_agents) * (-1 / n_agents)).cpu()
    h = torch.cat((h1, h2), 0)

    constrains = [A @ x == b, G @ x <= h]
    objective = cp.Minimize(cp.sum_squares(Q_sqrt_cvxpy @ x))
    problem = cp.Problem(objective, constrains)

    cvxpylayer = CvxpyLayer(problem, parameters=[Q_sqrt_cvxpy], variables=[x])

    try:
        solution, = cvxpylayer(Q ** 0.5)
        pos_weights = solution[: n_agents].to(device)
        weights[weights > 0.0] = pos_weights
    except:
        print("SolverError")
        print(vars)
        if L == 0.00001:
            print("turn to VarOpt")
            return var_opt_aggr(plosses)
        weights = error_opt_aggr(plosses, L=L*0.1)

    return weights

def aggr_batch(plosses_batch, method="OptAggr"):
    n_batch = plosses_batch.shape[0]
    n_agents = plosses_batch.shape[1]
    weights_ls = []

    if method == "VarOpt":
        for i in range(n_batch):
            weights_ls.append(var_opt_aggr(plosses_batch[i]).view(1, n_agents))
    elif method == "OptAggr":
        for i in range(n_batch):
            weights_ls.append(error_opt_aggr(plosses_batch[i]).view(1, n_agents))
    elif method == "ConvlAggr":
        for i in range(n_batch):
            weights_ls.append(data_size_aggr(plosses_batch[i]).view(1, n_agents))
    else:
        raise ValueError(f"{method} aggregation is not defined")

    weights_batch = torch.cat(weights_ls)

    return weights_batch


def error_bound_by_plosses_batch(plosses_batch, sensi, L, method="VarOpt", eval_mode=False):
    batch_size = plosses_batch.shape[0]
    n_agents = plosses_batch.shape[1]
    device = plosses_batch.device

    ls = []
    for batch in range(batch_size):
        plosses = plosses_batch[batch, :].view(n_agents)
        error_bound = error_bound_by_plosses(plosses, sensi, L, method, eval_mode=eval_mode)
        ls.append(error_bound)
    error_batch = torch.cat(ls, dim=0)

    return error_batch.view(batch_size, 1)


def error_bound_by_plosses(plosses, sensi, L, method='VarOpt', eval_mode=False):
    device = plosses.device
    n_agents = plosses.shape[0]
    nz_plosses = plosses[plosses > 0.]

    if eval_mode and nz_plosses.shape[0] == 0:
        return torch.tensor(float("inf"), device=device, dtype=plosses.dtype).view(1)

    if method == 'VarOpt':
        weights = var_opt_aggr(plosses)
    elif method == "OptAggr":
            weights = error_opt_aggr(plosses)
    elif method == 'ConvlAggr':
        weights = data_size_aggr(plosses)

    else:
        raise ValueError(f"{method} aggregation is not defined")

    if torch.isinf(weights).sum() > 0.0:
        print("weights contains inf")
        print(weights)

    if torch.isnan(weights).sum() > 0.0:
        print("weights contains nan")
        print(weights)


    var = ((weights[plosses > 0.0] * sensi / plosses[plosses > 0.])**2 * 2).sum()



    if torch.isinf(var).sum() > 0.0:
        print("var contains inf")
        print(var)

    if torch.isnan(var).sum() > 0.0:
        print("var contains nan")
        print(var)

    quad_bias = (torch.abs(weights - 1/n_agents)).sum() * L

    error_bound = quad_bias ** 2 + var
    return error_bound.view(1)


def error_bound_by_allocs_batch(allocs_batch, pbudgets_batch, sensi, L, method='VarOpt'):
    n_agents = allocs_batch.shape[1]
    n_items = allocs_batch.shape[2]
    device = allocs_batch.device

    deno = torch.arange(n_items, 0, step=-1, dtype=allocs_batch.dtype, device=device)
    items = pbudgets_batch.view(-1, n_agents, 1).repeat(1, 1, n_items)
    items = items / deno

    privacy_losses = torch.sum(allocs_batch * items, dim=2)
    error_batch = error_bound_by_plosses_batch(privacy_losses, sensi, L, method=method)

    return error_batch

