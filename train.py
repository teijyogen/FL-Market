import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
import torch
from torch.nn.parallel import DataParallel
import numpy as np
from datasets import generate_dataset_from_json
from regretnet import RegretNet, train_loop

from datasets import Dataloader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=1)
parser.add_argument('--num-examples', type=int, default=102400)
parser.add_argument('--test-num-examples', type=int, default=10000)
parser.add_argument('--n-agents', type=int, default=10)
parser.add_argument('--n-items', type=int, default=20)
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--test-batch-size', type=int, default=500)
parser.add_argument('--model-lr', type=float, default=0.001)
parser.add_argument('--max-pbudget', type=float, default=8.0)
parser.add_argument('--min-pbudget', type=float, default=2.0)

parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=100)
parser.add_argument('--test-misreport-iter', type=int, default=1000)

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
parser.add_argument('--p_activation', default='softmax')
parser.add_argument('--a_activation', default='softmax')
parser.add_argument('--hidden_layer_size', type=int, default=100)
parser.add_argument('--n_hidden_layers', type=int, default=2)
parser.add_argument('--separate', action='store_true', default=False)
parser.add_argument('--rs_loss', action='store_true')
parser.add_argument('--smoothing', type=float, default=0.125)
parser.add_argument('--normalized_loss', action='store_true', default=False)

parser.add_argument('--teacher-model', default="")
parser.add_argument('--name', default='test')

if __name__ == "__main__":
    args = parser.parse_args()
    args.separate = True
    args.n_agents = 10
    args.n_items = 20
    args.max_pbudget = 5.0
    args.min_pbudget = 0.5
    # args.aggr_method = "VarOpt"
    args.aggr_method = "ConvlAggr"
    args.a_activation = 'deterministic'

    train_data_file = "data/train_profiles.json"
    test_data_file = "data/test_profiles.json"

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23453', rank=0, world_size=1)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


    model = RegretNet(args.n_agents, args.n_items, activation='relu', hidden_layer_size=args.hidden_layer_size,
                      n_hidden_layers=args.n_hidden_layers, p_activation=args.p_activation,
                      a_activation=args.a_activation, separate=args.separate, smoothing=args.smoothing)
    model = DataParallel(model).cuda()

    train_data = generate_dataset_from_json(train_data_file, args.n_items).cuda()
    train_loader = Dataloader(train_data, batch_size=args.batch_size)
    test_data = generate_dataset_from_json(test_data_file, args.n_items).cuda()
    test_loader = Dataloader(test_data, batch_size=args.test_batch_size)

    train_loop(
        model, train_loader, test_loader, args, device=DEVICE
    )

