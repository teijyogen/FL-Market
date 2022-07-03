import torch
import numpy as np

def fairquery(reports, budget):
    n_agents = len(reports)

    ids = np.array([i for i in range(n_agents)])
    reports = np.insert(reports, 0, values=ids, axis=1)
    max_pbudget = reports[:, 2].max()
    reports_byval = np.array(sorted(reports, key=lambda x:x[1]))

    selections = np.zeros(n_agents)
    k = 0
    for i in range(n_agents):
        if reports_byval[i, 1] <= budget/(k + 1):
            if (reports_byval[:i+1, 2] >= max_pbudget/(n_agents-(k+1))).all():
                k += 1
                selections[int(reports_byval[i, 0])] = 1.0
            else:
                break
        else:
            break
    if k > 0:
        payment_for_winners = min(budget/k, reports_byval[k, 1])
    else:
        payment_for_winners = 0.0

    plosses = max_pbudget / (n_agents - k) * selections
    payments = payment_for_winners * selections

    return plosses, payments

def allin(reports, budget):
    n_agents = len(reports)

    ids = np.array([i for i in range(n_agents)])
    reports = np.insert(reports, 0, values=ids, axis=1)
    # print(reports)

    valuations = reports[:, 1]
    pbudgets = reports[:, 2]
    sizes = reports[:, 3]
    unit_prices = valuations / pbudgets / sizes

    reports_unitp = np.insert(reports, 4, values=unit_prices, axis=1)
    reports_unitp_byp = np.array(sorted(reports_unitp, key=lambda x:x[4]))

    selections = np.zeros(n_agents)
    accum_pbudget = 0.0
    for i in range(n_agents):
        pbudget = reports_unitp_byp[i, 2]
        size = reports_unitp_byp[i, 3]
        unit_price = reports_unitp_byp[i, 4]
        accum_pbudget_temp = accum_pbudget + pbudget * size

        if accum_pbudget_temp * unit_price <= budget:
            selections[int(reports_unitp_byp[i, 0])] = 1.0
            accum_pbudget = accum_pbudget_temp

    if accum_pbudget > 0.0:
        critical_price = budget / accum_pbudget
    else:
        critical_price = 0.0
    plosses = pbudgets * selections
    payments = plosses * critical_price * sizes

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
    plosses_batch = np.array(plosses_batch)
    payments_batch = np.array(payments_batch)
    return torch.tensor(plosses_batch, device=device).float(), torch.tensor(payments_batch, device=device).float()

def multi_to_single(multi_reports):
    single_reports = multi_reports[:, :, -3:]
    return single_reports









