import torch
import numpy as np

def fairquery(reports, budget):
    n_agents = len(reports)

    ids = np.array([(i + 1) for i in range(n_agents)])
    reports = np.insert(reports, 0, values=ids, axis=1)
    max_pbudget = 1.0
    reports_byval = np.array(sorted(reports, key=lambda x:x[1]))

    selections = np.zeros(n_agents)
    k = 0
    for i in range(n_agents):
        if k+1 == n_agents:
            break
        if reports_byval[i, 1] <= budget/(k+1):
            if (reports_byval[:i+1, 2] >= max_pbudget/(n_agents-(k+1))).all():
                k += 1
                selections[int(reports[i, 0]-1)] = 1.0
            else:
                break
        else:
            break
    if k > 0:
        critical_price = min(budget/k, reports_byval[k, 1])
    else:
        critical_price = 0.0

    plosses = max_pbudget / (n_agents - k) * selections
    payments = critical_price * selections

    return plosses, payments

def allin(reports, budget):
    n_agents = len(reports)

    ids = np.array([(i + 1) for i in range(n_agents)])
    reports = np.insert(reports, 0, values=ids, axis=1)
    # print(reports)

    valuations = reports[:, 1]
    pbudgets = reports[:, 2]
    unit_prices = valuations / pbudgets

    reports_unitp = np.insert(reports, 3, values=unit_prices, axis=1)
    reports_unitp_byp = np.array(sorted(reports_unitp, key=lambda x:x[3]))

    selections = np.zeros(n_agents)
    accum_pbudgets = 0.0
    for i in range(n_agents):
        accum_pbudgets_temp = accum_pbudgets + reports_unitp_byp[i, 2]
        unit_price = reports_unitp_byp[i, 3]
        if accum_pbudgets_temp * unit_price <= budget:
            selections[int(reports_unitp_byp[i, 0])-1] = 1.0
            accum_pbudgets = accum_pbudgets_temp
            continue
    if accum_pbudgets > 0.0:
        critical_price = budget / accum_pbudgets
    else:
        critical_price = 0.0
    plosses = reports[:, 2] * selections
    payments = plosses * critical_price

    return plosses, payments


def baseline_batch(reports_batch, budget_batch, method='All-in'):
    batch_size = reports_batch.shape[0]
    # print(reports_batch.shape)
    # print(budget_batch.shape)
    device = reports_batch.device
    reports_batch = multi_to_single(reports_batch)
    reports_batch = reports_batch.clone().detach().to("cpu").numpy()
    budget_batch = budget_batch.clone().detach().to("cpu").numpy()

    plosses_batch = []
    payments_batch = []
    for batch in range(batch_size):
        reports = reports_batch[batch]
        budget = budget_batch[batch]

        plosses = []
        payments = []
        if method == 'All-in':
            plosses, payments = allin(reports, budget)
        elif method == 'FairQuery':
            plosses, payments = fairquery(reports, budget)
        else:
            print("Undefined aggregation mechanism")
        plosses_batch.append(plosses)
        payments_batch.append(payments)
    return torch.tensor(plosses_batch, device=device), torch.tensor(payments_batch, device=device)

def multi_to_single(multi_reports):
    single_reports = multi_reports[:, :, -2:]
    return single_reports









