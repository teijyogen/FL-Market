import torch
import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler

LEGEND_FONT = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }

LABEL_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 18,
         }

TITLE_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 18,
         }


def calc_deter_violation(allocs):
    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    device= allocs.device

    full_allocs = calc_full_allocs(allocs)
    uni_allocs = (torch.ones(full_allocs.shape) * (1/(n_items+1))).to(device)
    dist = torch.linalg.norm(full_allocs - uni_allocs, dim=2) ** 2
    max_dist = ((1.0 - 1 / (n_items + 1)) ** 2 + (0.0 - 1 / (n_items + 1)) ** 2 * n_items)
    return max_dist - dist


def create_combined_misreports(misreports, valuations):
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2] - 1

    mask = torch.zeros(
        (misreports.shape[0], n_agents, n_agents, n_items+1), device=misreports.device
    )
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0

    tiled_mis = misreports.view(-1, 1, n_agents, n_items+1).repeat(1, n_agents, 1, 1)
    tiled_true = valuations.view(-1, 1, n_agents, n_items+1).repeat(1, n_agents, 1, 1)

    return mask * tiled_mis + (1.0 - mask) * tiled_true


def allocs_to_plosses(allocs, pbudgets):

    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    device = allocs.device
    deno = torch.arange(n_items, 0, step=-1, dtype=pbudgets.dtype, device=device)
    items = pbudgets.view(-1, n_agents, 1).repeat(1, 1, n_items)
    items = items / deno
    plosses = torch.sum(allocs * items, dim=2)
    return plosses


def allocs_instantiate_plosses(allocs, pbudgets):
    n_items = allocs.shape[2]
    n_agents = allocs.shape[1]
    n_batches = allocs.shape[0]
    device = allocs.device

    probs_of_zero_loss = torch.ones(n_batches, n_agents, 1, dtype=allocs.dtype, device=device) - allocs.sum(dim=2).view(n_batches, n_agents, 1)
    full_allocs = torch.cat([probs_of_zero_loss, allocs], dim=2)
    full_allocs = full_allocs.clamp_min_(min=0.0)
    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains nan")
        print(full_allocs)

    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains inf")
        print(full_allocs)

    results = torch.multinomial(full_allocs.view(-1, n_items + 1), 1).view(-1, n_agents)
    per_pbudgets = pbudgets.view(-1, n_agents) / n_items

    return per_pbudgets * results


def calc_full_allocs(allocs):
    n_agents = allocs.shape[1]
    n_batches = allocs.shape[0]
    n_items = allocs.shape[2]
    device = allocs.device

    probs_of_zero_loss = torch.ones(n_batches, n_agents, 1, dtype=allocs.dtype, device=device) - allocs.sum(dim=2).view(n_batches, n_agents, 1)
    full_allocs = torch.cat([probs_of_zero_loss, allocs], dim=2)
    full_allocs = full_allocs.clamp_min_(min=0.0)
    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains nan")
        print(full_allocs)

    if torch.isnan(full_allocs).sum() > 0.0:
        print("full_allocs contains inf")
        print(full_allocs)

    return full_allocs.view(-1, n_agents, n_items+1)


def calc_agent_util(reports, agent_allocations, payments):
    n_agents = payments.shape[1]

    valuations = reports[:, :, :-1]
    pbudgets = reports[:, :, -1].view(-1, n_agents)
    privacy_losses = allocs_to_plosses(agent_allocations, pbudgets)

    costs = torch.sum(agent_allocations * valuations, dim=2)
    util = payments - costs
    violations = (pbudgets - privacy_losses) < 0
    util = torch.where(violations, torch.zeros(violations.shape, device=violations.device), util)

    return util


def create_misreports(
    model, current_reports, current_misreports, budget, misreport_iter=10, lr=1e-1
):

    current_misreports.requires_grad_(True)
    true_pbudgets = current_reports[:, :, -1]

    for i in range(misreport_iter):
        model.zero_grad()
        agent_utils = tiled_misreport_util(current_misreports, current_reports, model, budget)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), current_misreports)
        misreports_grad = torch.where(torch.isnan(misreports_grad), torch.full_like(misreports_grad, 0), misreports_grad)

        with torch.no_grad():
            current_misreports += lr * misreports_grad
            model.clamp_op(current_misreports)

            fake_pbudgets = current_misreports[:, :, -1]
            fake_pbudgets[fake_pbudgets > true_pbudgets] = true_pbudgets[fake_pbudgets > true_pbudgets]

    return current_misreports


def optimize_misreports(
    model, current_reports, current_misreports, budget, misreport_iter=10, lr=1e-1
):

    current_misreports.requires_grad_(True)
    true_pbudgets = current_reports[:, :, -1]

    for i in range(misreport_iter):
        model.zero_grad()
        agent_utils = tiled_misreport_util(current_misreports, current_reports, model, budget)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), current_misreports)
        misreports_grad = torch.where(torch.isnan(misreports_grad), torch.full_like(misreports_grad, 0), misreports_grad)

        with torch.no_grad():
            current_misreports += lr * misreports_grad
            model.module.clamp_op(current_misreports)

            fake_pbudgets = current_misreports[:, :, -1]
            fake_pbudgets[fake_pbudgets > true_pbudgets] = true_pbudgets[fake_pbudgets > true_pbudgets]

    return current_misreports


def tiled_misreport_util(current_misreports, current_reports, model, budget):
    n_agents = current_reports.shape[1]
    n_items = current_reports.shape[2] - 1

    agent_idx = list(range(n_agents))
    tiled_misreports = create_combined_misreports(
        current_misreports, current_reports
    )
    flatbatch_tiled_misreports = tiled_misreports.view(-1, n_agents, n_items+1)
    tiled_budget = budget.repeat(n_agents, 1)
    allocations, payments = model((flatbatch_tiled_misreports, tiled_budget))
    reshaped_payments = payments.view(
        -1, n_agents, n_agents
    )
    reshaped_allocations = allocations.view(-1, n_agents, n_agents, n_items)
    agent_payments = reshaped_payments[:, agent_idx, agent_idx]
    agent_allocations = reshaped_allocations[:, agent_idx, agent_idx, :]
    agent_utils = calc_agent_util(
        current_reports, agent_allocations, agent_payments
    )
    return agent_utils


def plot(x_list, y_list, labels, title, file_name, xlabel, ylabel, yscale='linear'):
    if len(labels) == 4 and ylabel == 'model accuracy':
        market_color_ls = [
            ('s', 'r'),
            ('h', 'g'),
            ('d', 'b'),
            ('o', 'k'),
        ]
    elif len(labels) == 4:
        market_color_ls = [
            ('s', 'r'),
            ('h', 'g'),
            ('d', 'b', '--'),
            ('o', 'k'),
        ]
    elif len(labels) == 3:
        market_color_ls = [
            ('o', 'g'),
            ('d', 'b', '--'),
            ('s', 'r'),
        ]
    else:
        market_color_ls = [
            ('v', 'b'),
            ('h', 'g'),
            ('d', 'm'),
            ('^', 'r'),
            ('o', 'c'),
            ('s', 'k'),
        ]

    for i in range(len(x_list)):
        if len(market_color_ls[i]) > 2:
            plt.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1],
                     linestyle=market_color_ls[i][2], zorder=4-i, markerfacecolor='none')
        else:
            plt.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1], alpha=1.0)


    plt.yscale(yscale)
    plt.tick_params(labelsize=8)
    plt.legend(prop=LEGEND_FONT)
    plt.xlabel(xlabel, LABEL_FONT)
    plt.ylabel(ylabel, LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_budget_acc(x_list, y_list, labels, title, file_name, yscale='logit'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget rate', 'model accuracy', yscale=yscale)


def plot_budget_mse(x_list, y_list, labels, title, file_name, yscale='log'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget rate', 'finite error bound', yscale=yscale)


def plot_budget_invalid_rate(x_list, y_list, labels, title, file_name, yscale='linear'):
    plot(x_list, y_list, labels, title, file_name, 'financial budget rate', 'invalid gradient rate', yscale=yscale)


def plot_n_agents_mse(x_list, y_list, labels, title, file_name, yscale='log'):
    plot(x_list, y_list, labels, title, file_name, 'number of data owners', 'finite error bound', yscale=yscale)


def plot_bar(x_list, y_list, labels, title, file_name, xlabel, ylabel):
    pattern = ('//', '\\')

    tick_ls = []
    for n_items in x_list:
        tick_ls.append(str(n_items))

    cols = list(range(len(x_list)))

    for i in range(len(y_list)):
        plt.bar(cols, y_list[i], width=0.4, label=labels[i], tick_label=tick_ls, hatch=pattern[i])
        for j in range(len(cols)):
            cols[j] = cols[j] + 0.4

    plt.tick_params(labelsize=8)
    plt.legend(prop=LEGEND_FONT)
    plt.xlabel(xlabel, LABEL_FONT)
    plt.ylabel(ylabel, LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_n_items_mse(x_list, y_list, labels, title, file_name):
    plot_bar(x_list, y_list, labels, title, file_name, xlabel='M-Unit', ylabel='finite error bound')
