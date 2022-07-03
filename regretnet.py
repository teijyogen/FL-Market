import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm
import torch.nn.init
import ibp
import json
from aggregation import error_bound_by_allocs_batch as calc_error_bound
from utils import *


class RegretNet(nn.Module):
    def __init__(self, n_agents, n_items, hidden_layer_size=128, clamp_op=None, n_hidden_layers=2,
                 activation='tanh', p_activation=None, a_activation='softmax', separate=False, smoothing=0.1,
                 normalized_input=-1, deter_train=True):
        super(RegretNet, self).__init__()

        self.activation = activation
        if activation == 'tanh':
            self.act = ibp.Tanh
        else:
            self.act = ibp.ReLU
        self.clamp_opt = clamp_op
        if clamp_op is None:
            def cl(x):
                x.clamp_min_(0.0)
            self.clamp_op = cl
        else:
            self.clamp_op = clamp_op


        self.p_activation = p_activation
        self.a_activation = a_activation
        self.n_agents = n_agents
        self.n_items = n_items
        self.input_size = self.n_agents * (self.n_items + 2) + 1
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers=n_hidden_layers
        self.separate = separate
        self.deter_train = deter_train
        self.smoothing = smoothing
        self.normalized_input = normalized_input


        if self.a_activation in ['softmax', 'deterministic']:
            self.allocations_size = (self.n_agents) * (self.n_items + 1)
            self.allocation_head = [ibp.Linear(self.hidden_layer_size, self.allocations_size),
                                    ibp.View((-1, self.n_agents, self.n_items+1)),
                                    ibp.Softmax(dim=2),
                                    ibp.View_Cut()]
        else:
            raise ValueError(f"{self.a_activation} behavior is not defined")


        if p_activation == 'softmax':
            self.payments_size = self.n_agents + 1
            self.payment_head = [
                ibp.Linear(self.hidden_layer_size, self.payments_size), ibp.Softmax(dim=1)
            ]
        else:
            raise ValueError('payment activation behavior is not defined')

        if separate:
            self.nn_model = ibp.Sequential(
                *([ibp.Identity(),
                   # nn.BatchNorm1d(self.input_size)
                   ])
            )
            self.payment_head = [ibp.Linear(self.input_size, self.hidden_layer_size),
                                 # nn.BatchNorm1d(self.hidden_layer_size),
                                 self.act()] + \
                                [l for i in range(n_hidden_layers)
                                 for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size),
                                           # nn.BatchNorm1d(self.hidden_layer_size),
                                           self.act())] + \
                                self.payment_head

            self.payment_head = ibp.Sequential(*self.payment_head)
            self.allocation_head = [ibp.Linear(self.input_size, self.hidden_layer_size),
                                    # nn.BatchNorm1d(self.hidden_layer_size),
                                    self.act()] + \
                                   [l for i in range(n_hidden_layers)
                                    for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size),
                                              # nn.BatchNorm1d(self.hidden_layer_size),
                                              self.act())] + \
                                   self.allocation_head
            self.allocation_head = ibp.Sequential(*self.allocation_head)
        else:
            self.nn_model = ibp.Sequential(
                *([ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] +
                  [l for i in range(n_hidden_layers)
                   for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
            )
            self.allocation_head = ibp.Sequential(*self.allocation_head)
            self.payment_head = ibp.Sequential(*self.payment_head)

    def normalize(self, input):
        input = input.view(input.shape[0], -1)
        if self.normalized_input == 1:
            norm_input = (input - input[:, :-1].mean(dim=1, keepdims=True)) / (input[:, :-1].max(dim=1, keepdims=True)[0] - input[:, :-1].min(dim=1, keepdims=True)[0])
        elif self.normalized_input == 2:
            norm_input = (input - input.min(dim=1, keepdims=True)[0]) / (input[:, :-1].max(dim=1, keepdims=True)[0] - input[:, :-1].min(dim=1, keepdims=True)[0])
            norm_input = norm_input * 2. - 1.
        else:
            norm_input = input

        norm_input[torch.isnan(norm_input)] = 0.0
        norm_input[torch.isinf(norm_input)] = 0.0
        return norm_input

    def forward(self, input):
        # reports = input[0].view(-1, self.n_agents * (self.n_items + 2))
        # budget = input[1]
        # x = torch.cat((reports, budget), dim=1)
        # x = self.nn_model(x)
        # allocs = self.allocation_head(x)

        # reports = input[0].view(-1, self.n_agents, self.n_items + 2)
        # budget = input[1]
        # sizes = reports[:, :, -1]
        # avg_budget = (budget / sizes.sum(dim=1, keepdim=True)).view(-1, 1)
        # Ws = sizes / sizes.sum(dim=1, keepdims=True)
        # Ws = Ws.view(-1, self.n_agents, 1)
        # reports = torch.cat((reports[:, :, :-1], Ws), dim=2)
        # reports = reports.view(-1, self.n_agents * (self.n_items + 2))
        # x = torch.cat((reports, avg_budget), dim=1)
        # x = self.nn_model(x)
        # allocs = self.allocation_head(x)

        reports = input[0].view(-1, self.n_agents, self.n_items + 2)
        budget = input[1]
        n_batch = reports.shape[0]

        sizes = reports[:, :, -1]
        avg_budget = (budget / sizes.sum(dim=1, keepdim=True)).view(-1, 1)
        norm_sizes = self.normalize(sizes).view(-1, self.n_agents, 1)
        if torch.isnan(norm_sizes).sum() > 0 or torch.isinf(norm_sizes).sum() > 0:
            print("nomr_sizes")
            print(norm_sizes)

        norm_vals_budget = torch.cat((reports[:, :, :-2].reshape(n_batch, -1), avg_budget), dim=1)
        norm_vals_budget = self.normalize(norm_vals_budget)

        norm_vals = norm_vals_budget[:, :-1].view(-1, self.n_agents, self.n_items)
        norm_budget = norm_vals_budget[:, -1].view(-1, 1)
        if torch.isnan(norm_vals).sum() > 0 or torch.isinf(norm_vals).sum() > 0:
            print("norm_vals")
            print(norm_sizes)

        if torch.isnan(norm_budget).sum() > 0 or torch.isinf(norm_budget).sum() > 0:
            print("norm_budget")
            print(norm_sizes)

        norm_pbudget = reports[:, :, -2]
        norm_pbudget = self.normalize(norm_pbudget).view(-1, self.n_agents, 1)
        if torch.isnan(norm_pbudget).sum() > 0 or torch.isinf(norm_pbudget).sum() > 0:
            print("norm_pbudget")
            print(norm_sizes)

        norm_reports = torch.cat((norm_vals, norm_pbudget, norm_sizes), dim=2)
        norm_reports = norm_reports.view(-1, self.n_agents * (self.n_items + 2))
        x = torch.cat((norm_reports, norm_budget), dim=1)

        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        if self.a_activation == 'deterministic':
            if self.deter_train:
                allocs = torch.nn.functional.softmax(allocs/self.smoothing, dim=2)
            else:
                shape = allocs.shape
                dtype = allocs.dtype
                hot = torch.argmax(allocs, dim=2).view(shape[0], shape[1], 1)
                allocs = torch.nn.functional.one_hot(hot, num_classes=shape[2]).float().view(shape[0], shape[1], shape[2])

        if self.p_activation == 'softmax':
            budget_rate = self.payment_head(x)[:, :-1]
            payments = budget_rate * budget
        else:
            payments = self.payment_head(x)

        return allocs, payments


def test_loop(
    model,
    loader,
    args,
    device='cpu'
):
    model.eval()
    model.module.deter_train = False
    total_regret = 0.0
    n_agents = model.module.n_agents
    n_items = model.module.n_items
    total_regret_by_agt = [0. for i in range(n_agents)]
    total_regret_sq_by_agt = [0. for i in range(n_agents)]

    total_error_loss = 0.0
    total_ir_violation = 0.0
    total_bc_violation = 0.0
    total_deter_violation = 0.0

    total_ir_violation_count = 0
    total_bc_violation_count = 0

    ir_violation_max = 0.0
    bc_violation_max = 0.0
    deter_violation_max = 0.0
    regret_max = 0.0

    total_regret_norm = 0.0
    total_ir_violation_norm = 0.0
    regret_norm_max = 0.0
    ir_violation_norm_max = 0.0

    n_count = 0
    print(args)

    for i, batch in tqdm(enumerate(loader)):

        batch = batch.to(device)
        val_type = batch[:, :, -2:]
        batch = batch[:, :, :-2]

        sizes = batch[:, :, -1].view(-1, args.n_agents)
        # critical_budget = generate_critical_budget(batch)
        # budget_rate = (torch.rand(critical_budget.shape).to(device) * (0.01 - 1.) + 1.)
        # budget = critical_budget * budget_rate

        budget = (batch[:, :, -3].view(-1, args.n_agents) * sizes).sum(dim=1).view(-1, 1)
        budget_rate = (torch.rand(budget.shape).to(device) * (args.min_budget_rate - args.max_budget_rate) + args.max_budget_rate)
        budget = budget * budget_rate


        misreport_batch = batch.clone().detach()
        n_count += batch.shape[0]

        optimize_misreports(model, batch, misreport_batch, budget, val_type, misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model((batch, budget))

        truthful_util = calc_agent_util(batch, allocs, payments, instantiation=True)
        misreport_util = tiled_misreport_util(misreport_batch, batch, model, budget, val_type, instantiation=True)

        pbudgets = batch[:, :, -2].view(-1, args.n_agents)
        errors = calc_error_bound(allocs, pbudgets, sizes, args.L, method=args.aggr_method)
        error_loss = errors.sum()
        total_error_loss += error_loss.item()

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)
        norm_regrets = positive_regrets / payments
        total_regret += positive_regrets.sum().item()
        total_regret_norm += norm_regrets.sum().item()

        for i in range(n_agents):
            total_regret_by_agt[i] += positive_regrets[:, i].sum().item()
            total_regret_sq_by_agt[i] += (positive_regrets[:, i]**2).sum().item()

        ir_violations = -torch.clamp(truthful_util, max=0)
        norm_ir_violations = ir_violations / payments
        total_ir_violation += ir_violations.sum().item()
        total_ir_violation_count += (truthful_util < 0.0).sum().item()
        total_ir_violation_norm += norm_ir_violations.sum().item()

        total_bc_violation += torch.clamp(payments.sum(dim=1) - budget.sum(dim=1), min=0).sum().item()
        total_bc_violation_count += (payments.sum(dim=1) > budget.sum(dim=1)).sum().item()

        deter_violations = calc_deter_violation(allocs)
        total_deter_violation += deter_violations.sum().item()

        if ir_violation_max < ir_violations.max():
            ir_violation_max = ir_violations.max().item()

        if ir_violation_norm_max < norm_ir_violations.max():
            ir_violation_norm_max = norm_ir_violations.max().item()

        if bc_violation_max < torch.clamp_min(payments.sum(dim=1) - budget.sum(dim=1), 0).max():
            bc_violation_max = torch.clamp_min(payments.sum(dim=1) - budget.sum(dim=1), 0).max().item()

        if regret_max < positive_regrets.max():
            regret_max = positive_regrets.max().item()

        if regret_norm_max < norm_regrets.max():
            regret_norm_max = norm_regrets.max().item()

        if deter_violation_max < deter_violations.sum().max():
            deter_violation_max = deter_violations.sum().max().item()

    result = {
              "error_loss": total_error_loss / n_count,
              # "regret_std": (total_regret_sq/n_count - (total_regret/n_count)**2)**.5,
              "regret_mean": total_regret/n_count/n_agents,
              "norm_regret_mean": total_regret_norm / n_count / n_agents,
              "regret_max": regret_max,
              "norm_regret_max": regret_norm_max,
              "ir_violation_mean": total_ir_violation/n_count/n_agents,
              "norm_ir_violation_mean": total_ir_violation_norm / n_count / n_agents,
              "ir_violation_count": total_ir_violation_count/n_count,
              # "ir_violation_std": (total_ir_violation_sq/n_count - (total_ir_violation/n_count)**2)**.5,
              "ir_violation_max": ir_violation_max,
              "norm_ir_violation_max": ir_violation_norm_max,
              "bc_violation_max": bc_violation_max,
              "bc_violation_mean": total_bc_violation/n_count,
              "bc_violation_count": total_bc_violation_count/n_count,
              "deter_violation_max": deter_violation_max,
              "deter_violation_mean": total_deter_violation / n_count,
              }
    for i in range(n_agents):
        result[f"regret_agt{i}_std"] = (total_regret_sq_by_agt[i]/n_count - (total_regret_by_agt[i]/n_count)**2)**.5
        result[f"regret_agt{i}_mean"] = total_regret_by_agt[i]/n_count
    model.module.deter_train = True
    model.train()

    return result


def train_loop(
    model,
    train_loader,
    test_loader,
    args,
    device="cpu",
    writer=None
):
    model.train()
    # budget_std = args.budget
    n_agents = model.module.n_agents
    n_items = model.module.n_items

    regret_lagr_mults = args.regret_lagr_mult * torch.ones((1, n_agents)).to(device)
    ir_lagr_mults = args.ir_lagr_mult * torch.ones((1, n_agents)).to(device)
    bc_lagr_mult = args.bc_lagr_mult
    deter_lagr_mults = args.deter_lagr_mult * torch.ones((1, n_agents)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)


    iter=0

    lagr_update_iter_regret = args.lagr_update_iter_regret
    lagr_update_iter_ir = args.lagr_update_iter_ir
    lagr_update_iter_bc = args.lagr_update_iter_bc
    lagr_update_iter_deter = args.lagr_update_iter_deter

    rho_regret = args.rho_regret
    rho_ir = args.rho_ir
    rho_bc = args.rho_bc
    rho_deter = args.rho_deter

    print_step = int(args.num_examples / args.batch_size / 10)

    for epoch in tqdm(range(args.num_epochs)):
        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            val_type = batch[:, :, -2:]
            batch = batch[:, :, :-2]

            misreport_batch = batch.clone().detach().to(device)

            sizes = batch[:, :, -1].view(-1, args.n_agents)
            # critical_budget = generate_critical_budget(batch)
            # budget_rate = (torch.rand(critical_budget.shape).to(device) * (0.01 - 1.) + 1.)
            # budget = critical_budget * budget_rate
            budget = (batch[:, :, -3].view(-1, args.n_agents) * sizes).sum(dim=1).view(-1, 1)
            budget_rate = (torch.rand(budget.shape).to(device) * (args.min_budget_rate - args.max_budget_rate) + args.max_budget_rate)
            budget = budget * budget_rate
            pbudgets = batch[:, :, -2].view(-1, args.n_agents)

            optimize_misreports(model, batch, misreport_batch, budget, val_type, misreport_iter=args.misreport_iter, lr=args.misreport_lr)
            # print(batch)
            allocs, payments = model((batch, budget))

            truthful_util = calc_agent_util(batch, allocs, payments)
            misreport_util = tiled_misreport_util(misreport_batch, batch, model, budget, val_type)
            costs = torch.sum(allocs * batch[:, :, :-2], dim=2) * sizes

            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)


            ir_violation = -torch.clamp(truthful_util, max=0)
            # bc_violation = torch.clamp(payments.sum(dim=1) - budget.sum(dim=1), min=0).mean()
            deter_violation = calc_deter_violation(allocs)


            #calculate losses
            errors = calc_error_bound(allocs, pbudgets, sizes, args.L, method=args.aggr_method)
            # total_plosses = allocs_to_plosses(allocs, pbudgets).sum(dim=1)
            plosses = allocs_to_plosses(allocs, pbudgets)
            weighted_plosses = (plosses * sizes) / sizes.mean(dim=1, keepdims=True)
            # weighted_pbudgets = pbudgets * sizes

            if torch.isnan(errors).sum() > 0.0:
                print("errors contains nan")

            if torch.isinf(errors).sum() > 0.0:
                print("errors contains inf")


            error_loss = errors[torch.isfinite(errors)].mean()

            if args.normalized_loss == 1:
                regret_loss = (regret_lagr_mults * positive_regrets / sizes.sum(dim=1, keepdims=True)).mean()
                regret_quad_loss = (rho_regret / 2.0) * ((positive_regrets / sizes.sum(dim=1, keepdims=True)) ** 2).mean()
                ir_loss = (ir_lagr_mults * ir_violation / sizes.sum(dim=1, keepdims=True)).mean()
                ir_quad_loss = (rho_ir / 2.0) * ((ir_violation / sizes.sum(dim=1, keepdims=True)) ** 2).mean()
                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()
            elif args.normalized_loss == 2:
                regret_loss = (regret_lagr_mults * positive_regrets / payments)
                regret_loss = regret_loss[torch.isfinite(regret_loss)].mean()

                regret_quad_loss = (rho_regret / 2.0) * ((positive_regrets / payments) ** 2)
                regret_quad_loss = regret_quad_loss[torch.isfinite(regret_quad_loss)].mean()

                ir_loss = (ir_lagr_mults * ir_violation / payments)
                ir_loss = ir_loss[torch.isfinite(ir_loss)].mean()

                ir_quad_loss = (rho_ir / 2.0) * ((ir_violation / payments) ** 2)
                ir_quad_loss = ir_quad_loss[torch.isfinite(ir_quad_loss)].mean()

                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()
            elif args.normalized_loss == 3:
                regret_loss = (regret_lagr_mults * positive_regrets / sizes).mean()
                regret_quad_loss = (rho_regret / 2.0) * ((positive_regrets / sizes) ** 2).mean()
                ir_loss = (ir_lagr_mults * ir_violation / sizes).mean()
                ir_quad_loss = (rho_ir / 2.0) * ((ir_violation / sizes) ** 2).mean()
                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()
            elif args.normalized_loss == 4:
                regret_loss = (regret_lagr_mults * positive_regrets / costs)
                regret_loss = regret_loss[torch.isfinite(regret_loss)].mean()

                regret_quad_loss = (rho_regret / 2.0) * ((positive_regrets / costs) ** 2)
                regret_quad_loss = regret_quad_loss[torch.isfinite(regret_quad_loss)].mean()

                ir_loss = (ir_lagr_mults * ir_violation / payments)
                ir_loss = ir_loss[torch.isfinite(ir_loss)].mean()

                ir_quad_loss = (rho_ir / 2.0) * ((ir_violation / payments) ** 2)
                ir_quad_loss = ir_quad_loss[torch.isfinite(ir_quad_loss)].mean()

                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()
            elif args.normalized_loss == 5:
                regret_loss = (regret_lagr_mults * positive_regrets / costs)
                regret_loss = regret_loss[torch.isfinite(regret_loss)].mean()

                regret_quad_loss = (rho_regret / 2.0) * ((positive_regrets / costs) ** 2)
                regret_quad_loss = regret_quad_loss[torch.isfinite(regret_quad_loss)].mean()

                ir_loss = (ir_lagr_mults * ir_violation / costs)
                ir_loss = ir_loss[torch.isfinite(ir_loss)].mean()

                ir_quad_loss = (rho_ir / 2.0) * ((ir_violation / costs) ** 2)
                ir_quad_loss = ir_quad_loss[torch.isfinite(ir_quad_loss)].mean()

                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()
            else:
                regret_loss = (regret_lagr_mults * positive_regrets).mean()
                regret_quad_loss = (rho_regret / 2.0) * (positive_regrets ** 2).mean()
                ir_loss = (ir_lagr_mults * ir_violation).mean()
                ir_quad_loss = (rho_ir / 2.0) * (ir_violation ** 2).mean()
                deter_loss = (deter_lagr_mults * deter_violation).mean()
                deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()


            if i % print_step == print_step - 1:
                print("\n------------------------------------------------------------------")
                print("Batch " + str(i+1) + ", Epoch " + str(epoch+1))
                print("------------------------------------------------------------------")
                print("error loss")
                print(error_loss)
                print("regret violation")
                print(positive_regrets.mean())
                print("ir violation")
                print(ir_violation.mean())
                print("regret violation perc. (utility)")
                print(positive_regrets.mean() / truthful_util.mean())
                print("regret violation perc. (cost)")
                print(positive_regrets.mean() / costs.mean())
                print("regret violation perc. (payment)")
                print(positive_regrets.mean() / payments.mean())
                print("ir violation perc.")
                print(ir_violation.mean() / payments.mean())
                print("norm. regret violation (utility)")
                print((positive_regrets / truthful_util.abs()).mean())
                print("norm. regret violation (cost)")
                print((positive_regrets / costs).mean())
                print("norm. regret violation (payment)")
                print((positive_regrets / payments).mean())
                print("norm. ir violation (utility)")
                print((ir_violation / truthful_util.abs()).mean())
                print("norm. ir violation (cost)")
                print((ir_violation / costs).mean())
                print("norm. ir violation (payment)")
                print((ir_violation / payments).mean())
                # print("bc violation")
                # print(bc_violation)
                print("deter violation")
                print(deter_violation.mean())
                print("privacy loss")
                print(plosses.sum(dim=1).mean())
                print("weighted privacy loss")
                print(weighted_plosses.sum(dim=1).mean())
                print("------------------------------------------------------------------")

            if args.a_activation == 'deterministic':
                loss_func = error_loss + regret_loss + regret_quad_loss + ir_loss + ir_quad_loss + deter_loss + deter_quad_loss
            else:
                loss_func = -weighted_plosses.mean() + regret_loss + regret_quad_loss + ir_loss + ir_quad_loss


            if torch.isnan(loss_func).sum() > 0.0:
                print("Batch " + str(i + 1) + ", Epoch " + str(epoch + 1))
                print("loss_func contains nan")
                print(loss_func)

            if torch.isinf(loss_func).sum() > 0.0:
                print("Batch " + str(i + 1) + ", Epoch " + str(epoch + 1))
                print("loss_func contains inf")
                print(loss_func)


            #update model
            optimizer.zero_grad()
            loss_func.backward()

            drop = False

            for param in model.parameters():
                if torch.isnan(param.grad).sum() > 0.0:
                    print("Batch " + str(i + 1) + ", Epoch " + str(epoch + 1))
                    print("param.grad contains nan")
                    print(param.grad)
                    drop = True
                    break

                if torch.isinf(param.grad).sum() > 0.0:
                    print("Batch " + str(i + 1) + ", Epoch " + str(epoch + 1))
                    print("param.grad contains inf")
                    print(param.grad)
                    drop = True
                    break

            if drop:
                optimizer.zero_grad()
            else:
                optimizer.step()

            #update various fancy multipliers
            if args.normalized_loss == 2:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets / payments, dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation / payments, dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)
            elif args.normalized_loss == 4:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets / costs, dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation / payments, dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)
            elif args.normalized_loss == 5:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets / costs, dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation / costs, dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)
            elif args.normalized_loss == 3:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets / sizes, dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation / sizes, dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)
            elif args.normalized_loss == 1:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets / sizes.sum(dim=1, keepdims=True), dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation / sizes.sum(dim=1, keepdims=True), dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)
            else:
                if iter % lagr_update_iter_regret == 0:
                    with torch.no_grad():
                        regret_lagr_mults += rho_regret * torch.mean(positive_regrets, dim=0)
                if iter % lagr_update_iter_ir == 0:
                    with torch.no_grad():
                        ir_lagr_mults += rho_ir * torch.mean(ir_violation, dim=0)
                # if iter % lagr_update_iter_bc == 0:
                #     with torch.no_grad():
                #         bc_lagr_mult += rho_bc * bc_violation
                if iter % lagr_update_iter_deter == 0:
                    with torch.no_grad():
                        deter_lagr_mults += rho_deter * torch.mean(deter_violation, dim=0)



        if epoch % args.rho_incr_epoch_regret == 0:
            rho_regret += args.rho_incr_amount_regret
        if epoch % args.rho_incr_epoch_ir == 0:
            rho_ir += args.rho_incr_amount_ir
        if epoch % args.rho_incr_epoch_deter == 0:
            rho_deter += args.rho_incr_amount_deter

        if epoch % 10 == 9:
            if test_loader:
                test_result = test_loop(model, test_loader, args, device=device)
                print(f"Epoch {str(epoch)}")
                print(json.dumps(test_result, indent=4, sort_keys=True))

                for key, value in test_result.items():
                    writer.add_scalar(f"test/stat/{key}", value, global_step=epoch)
            arch = {'n_agents': model.module.n_agents,
                    'n_items': model.module.n_items,
                    'hidden_layer_size': model.module.hidden_layer_size,
                    'n_hidden_layers': model.module.n_hidden_layers,
                    'clamp_op': model.module.clamp_opt,
                    'activation': model.module.activation,
                    'p_activation': model.module.p_activation,
                    'a_activation': model.module.a_activation,
                    'separate': model.module.separate,
                    "normalized_input": model.module.normalized_input}
            torch.save({'name': args.name,
                        'arch': arch,
                        'state_dict': model.cpu().state_dict(),
                        'args': args}
                       , f"result/{args.name}_{epoch+1}_checkpoint.pt")

            model = model.to(device)


