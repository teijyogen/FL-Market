import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--num-examples', type=int, default=102400)
parser.add_argument('--test-num-examples', type=int, default=10000)
parser.add_argument('--n-agents', type=int, default=10)
parser.add_argument('--n-items', type=int, default=20)
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--test-batch-size', type=int, default=500)
parser.add_argument('--model-lr', type=float, default=0.001)
parser.add_argument('--max-budget-rate', type=float, default=5.0)
parser.add_argument('--min-budget-rate', type=float, default=0.1)
parser.add_argument('--activation', default="tanh")

parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=100)
parser.add_argument('--test-misreport-iter', type=int, default=100)

parser.add_argument('--rho-regret', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-regret', type=int, default=1)
parser.add_argument('--rho-incr-amount-regret', type=float, default=1.0)

parser.add_argument('--rho-ir', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-ir', type=int, default=1)
parser.add_argument('--rho-incr-amount-ir', type=float, default=1.0)

parser.add_argument('--rho-bc', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-bc', type=int, default=1)
parser.add_argument('--rho-incr-amount-bc', type=float, default=1.0)

parser.add_argument('--rho-deter', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-deter', type=int, default=1)
parser.add_argument('--rho-incr-amount-deter', type=float, default=.0)

parser.add_argument('--regret-lagr-mult', type=float, default=1.0)
parser.add_argument('--ir-lagr-mult', type=float, default=1.0)
parser.add_argument('--bc-lagr-mult', type=float, default=1.0)
parser.add_argument('--deter-lagr-mult', type=float, default=1.0)

parser.add_argument('--lagr-update-iter-regret', type=int, default=10)
parser.add_argument('--lagr-update-iter-ir', type=int, default=10)
parser.add_argument('--lagr-update-iter-bc', type=int, default=10)
parser.add_argument('--lagr-update-iter-deter', type=int, default=10)

parser.add_argument('--resume', default="")
parser.add_argument('--sensi', type=float, default=2.0)
parser.add_argument('--L', type=float, default=1.0)
parser.add_argument('--aggr-method', type=str, default="OptAggr")

#architectural arguments
parser.add_argument('--p-activation', default='softmax')
parser.add_argument('--a-activation', default='softmax')
parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--separate', default=True)
parser.add_argument('--smoothing', type=float, default=0.125)
parser.add_argument('--normalized-loss', type=int, default=-1)
parser.add_argument('--normalized-input', type=int, default=-1)

parser.add_argument('--teacher-model', default="")
parser.add_argument('--name', default='test')
parser.add_argument('--host', type=int, default=23456)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--data-dir', default='data/')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from torch.nn.parallel import DataParallel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from datasets import generate_dataset_from_json
from regretnet import RegretNet, train_loop, test_loop
from datasets import Dataloader
from client import Clients
import json

if __name__ == "__main__":

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:%s' %(args.host), rank=0, world_size=1)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    model = RegretNet(args.n_agents, args.n_items, activation=args.activation, hidden_layer_size=args.hidden_layer_size,
                      n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                      a_activation=args.a_activation, separate=args.separate, smoothing=args.smoothing, normalized_input=args.normalized_input)
    model = DataParallel(model).cuda()
    # model_name = "result/kn0529_dm_opt_nslkdd_iid_50_checkpoint.pt"
    # model_dict = torch.load(model_name)
    # arch = model_dict["arch"]
    # state_dict = model_dict["state_dict"]
    # model = RegretNet(arch["n_agents"], arch["n_items"], activation=arch["activation"], hidden_layer_size=arch["hidden_layer_size"],
    #                   n_hidden_layers=arch["n_hidden_layers"], p_activation=arch["p_activation"],
    #                   a_activation=arch["a_activation"], separate=arch["separate"])
    # model = DataParallel(model)
    # model.load_state_dict(state_dict)

    clients = Clients()
    clients.dirs = args.data_dir

    clients.filename = "train_profiles_2mp.json"
    clients.load_json()
    train_data = torch.tensor(clients.return_bids(args.n_items)).float().cuda().view(-1, args.n_agents, args.n_items + 4)
    print(train_data.shape)
    train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)

    # clients.filename = "test_profiles_2mp.json"
    # clients.load_json()
    # test_data = torch.tensor(clients.return_bids(args.n_items)).float().cuda().view(-1, args.n_agents, args.n_items + 4)
    # test_data = test_data[:args.test_num_examples]
    # test_loader = Dataloader(test_data, batch_size=args.test_batch_size)

    train_loop(
        model, train_loader, None, args, device=DEVICE
    )

    # test_result = test_loop(
    #     model, test_loader, args, device=DEVICE
    # )
    # print(json.dumps(test_result, indent=4, sort_keys=True))