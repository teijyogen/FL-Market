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
                 activation='tanh', p_activation=None, a_activation='softmax', separate=False, train=True, smoothing=0.1):
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
        self.input_size = self.n_agents * (self.n_items + 1) + 1
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers=n_hidden_layers
        self.separate = separate
        self.train = train
        self.smoothing = smoothing


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
                *([ibp.Identity()])
            )
            self.payment_head = [ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                [l for i in range(n_hidden_layers)
                                 for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                self.payment_head

            self.payment_head = ibp.Sequential(*self.payment_head)
            self.allocation_head = [ibp.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                   [l for i in range(n_hidden_layers)
                                    for l in (ibp.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
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


    def forward(self, input):
        reports = input[0].view(-1, self.n_agents * (self.n_items + 1))
        budget = input[1]
        x = torch.cat((reports, budget), dim=1)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        if self.a_activation == 'deterministic':
            if self.train:
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

    n_count = 0
    print(args)

    for i, batch in tqdm(enumerate(loader)):

        batch = batch.to(device)

        budget = batch[:, :, -2].view(-1, args.n_agents).sum(dim=1).view(-1, 1)
        budget = budget * (torch.rand(budget.shape).to(device) * (0.01 - 1.0) + 1.0)

        misreport_batch = batch.clone().detach()
        n_count += batch.shape[0]

        optimize_misreports(model, batch, misreport_batch, budget, misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model((batch, budget))

        payments_limit = torch.sum(
            (allocs * batch[:, :, :-1]).view(batch.shape[0], args.n_agents, args.n_items), dim=2
        )

        truthful_util = calc_agent_util(batch, allocs, payments)
        misreport_util = tiled_misreport_util(misreport_batch, batch, model, budget)

        pbudgets = batch[:, :, -1].view(-1, args.n_agents)
        errors = calc_error_bound(allocs, pbudgets, args.sensi, args.L, method=args.aggr_method)
        error_loss = errors.sum()
        total_error_loss += error_loss.item()

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)
        total_regret += positive_regrets.sum().item()

        for i in range(n_agents):
            total_regret_by_agt[i] += positive_regrets[:, i].sum().item()
            total_regret_sq_by_agt[i] += (positive_regrets[:, i]**2).sum().item()

        total_ir_violation += torch.clamp_min(payments - payments_limit, 0).sum().item()
        total_ir_violation_count += (payments > payments_limit).sum().item()

        total_bc_violation += torch.clamp(payments.sum(dim=1) - budget.sum(dim=1), min=0).sum().item()
        total_bc_violation_count += (payments.sum(dim=1) > budget.sum(dim=1)).sum().item()

        total_deter_violation += calc_deter_violation(allocs).sum().item()

        if ir_violation_max < torch.clamp_min(payments - payments_limit, 0).max():
            ir_violation_max = torch.clamp_min(payments - payments_limit, 0).max().item()

        if bc_violation_max < torch.clamp_min(payments.sum(dim=1) - budget.sum(dim=1), 0).max():
            bc_violation_max = torch.clamp_min(payments.sum(dim=1) - budget.sum(dim=1), 0).max().item()

        if regret_max < torch.clamp_min(regrets, 0).max():
            regret_max = torch.clamp_min(regrets, 0).max().item()

        if deter_violation_max < calc_deter_violation(allocs).sum().max():
            ir_violation_max = calc_deter_violation(allocs).sum().max().item()

    result = {
              "error_loss": total_error_loss / n_count,
              # "regret_std": (total_regret_sq/n_count - (total_regret/n_count)**2)**.5,
              "regret_mean": total_regret/n_count/n_agents,
              "regret_max": regret_max,
              "ir_violation_mean": total_ir_violation/n_count/n_agents,
              "ir_violation_count": total_ir_violation_count/n_count,
              # "ir_violation_std": (total_ir_violation_sq/n_count - (total_ir_violation/n_count)**2)**.5,
              "ir_violation_max": ir_violation_max,
              "bc_violation_max": bc_violation_max,
              "bc_violation_mean": total_bc_violation/n_count,
              "bc_violation_count": total_bc_violation_count/n_count,
              "deter_violation_max": deter_violation_max,
              "deter_violation_mean": total_deter_violation / n_count,
              }
    for i in range(n_agents):
        result[f"regret_agt{i}_std"] = (total_regret_sq_by_agt[i]/n_count - (total_regret_by_agt[i]/n_count)**2)**.5
        result[f"regret_agt{i}_mean"] = total_regret_by_agt[i]/n_count
    return result


def train_loop(
    model,
    train_loader,
    test_loader,
    args,
    device="cpu",
    writer=None
):
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

    for epoch in tqdm(range(args.num_epochs)):
        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            batch_size = batch.shape[0]
            n_agents = batch.shape[1]
            n_items = batch.shape[2]
            misreport_batch = batch.clone().detach().to(device)

            budget = batch[:, :, -2].view(-1, args.n_agents).sum(dim=1).view(-1, 1)
            budget_rate = (torch.rand(budget.shape).to(device) * (0.01 - 1.0) + 1.0)
            budget = budget * budget_rate


            optimize_misreports(model, batch, misreport_batch, budget, misreport_iter=args.misreport_iter, lr=args.misreport_lr)
            # print(batch)
            allocs, payments = model((batch, budget))

            truthful_util = calc_agent_util(batch, allocs, payments)
            misreport_util = tiled_misreport_util(misreport_batch, batch, model, budget)


            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)


            ir_violation = -torch.clamp(truthful_util, max=0)
            bc_violation = torch.clamp(payments.sum(dim=1) - budget.sum(dim=1), min=0).mean()
            deter_violation = calc_deter_violation(allocs)


            #calculate losses
            pbudgets = batch[:, :, -1].view(-1, args.n_agents)
            errors = calc_error_bound(allocs, pbudgets, args.sensi, args.L, method=args.aggr_method)
            total_plosses = allocs_to_plosses(allocs, pbudgets).sum(dim=1)

            if torch.isnan(errors).sum() > 0.0:
                print("errors contains nan")

            if torch.isinf(errors).sum() > 0.0:
                print("errors contains inf")


            error_loss = errors[torch.isfinite(errors)].mean()



            regret_loss = (regret_lagr_mults * positive_regrets).mean()
            regret_quad_loss = (rho_regret / 2.0) * (positive_regrets ** 2).mean()
            ir_loss = (ir_lagr_mults * ir_violation).mean()
            ir_quad_loss = (rho_ir / 2.0) * (ir_violation ** 2).mean()
            deter_loss = (deter_lagr_mults * deter_violation).mean()
            deter_quad_loss = (rho_deter / 2.0) * (deter_violation ** 2).mean()


            if i % 10 == 9:
                print("\n------------------------------------------------------------------")
                print("Batch " + str(i+1) + ", Epoch " + str(epoch+1))
                print("------------------------------------------------------------------")
                print("error loss")
                print(errors.mean())
                print("regret violation")
                print(positive_regrets.mean())
                print("ir violation")
                print(ir_violation.mean())
                print("bc violation")
                print(bc_violation)
                print("deter violation")
                print(deter_violation.mean())
                print("------------------------------------------------------------------")

            if args.a_activation == 'deterministic':
                loss_func = error_loss + regret_loss + regret_quad_loss + ir_loss + ir_quad_loss + deter_loss + deter_quad_loss
            else:
                loss_func = -total_plosses.mean() + regret_loss + regret_quad_loss + ir_loss + ir_quad_loss


            if torch.isnan(loss_func).sum() > 0.0:
                print("loss_func contains nan")
                print(loss_func)

            if torch.isinf(loss_func).sum() > 0.0:
                print("loss_func contains inf")
                print(loss_func)


            #update model
            optimizer.zero_grad()
            loss_func.backward()

            drop = False

            for param in model.parameters():
                if torch.isnan(param.grad).sum() > 0.0:
                    print("param.grad contains nan")
                    print(param.grad)
                    drop = True
                    break

                if torch.isinf(param.grad).sum() > 0.0:
                    print("param.grad contains inf")
                    print(param.grad)
                    drop = True
                    break

            if drop:
                optimizer.zero_grad()
            else:
                optimizer.step()

            #update various fancy multipliers
            if iter % lagr_update_iter_regret == 0:
                with torch.no_grad():
                    regret_lagr_mults += rho_regret * torch.mean(positive_regrets, dim=0)
            if iter % lagr_update_iter_ir == 0:
                with torch.no_grad():
                    ir_lagr_mults += rho_ir * torch.mean(ir_violation, dim=0)
            if iter % lagr_update_iter_bc == 0:
                with torch.no_grad():
                    bc_lagr_mult += rho_bc * bc_violation
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
                    'separate': model.module.separate}
            torch.save({'name': args.name,
                        'arch': arch,
                        'state_dict': model.state_dict(),
                        'args': args}
                       , f"result/{args.name}_{epoch+1}_checkpoint.pt")




