import distutils.util
import os
import torch
import random
import argparse
import numpy as np
from datasets import *
from config import _C as C
from torch.utils.data import DataLoader
from models.FGNS import FGNS
from evaluator import PredEvaluator
from collections import OrderedDict
from datasets.slide_same_r_dataset import SlideSameRDataset
from datasets.slide_dataset import SlideDataset
from datasets.sand_dataset import SandDataset
from datasets.crash_dataset import CrashDataset
from datasets.slide_plus_dataset import SlidePlusDataset
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='Eval parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--ckpt', type=str, help='', default=None)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--dataset-name', type=str, default="Slide_Plus", help='dataset name')
    parser.add_argument('--pred-all', type=lambda x:bool(distutils.util.strtobool(x)), default=True)
    parser.add_argument('--output-gif', type=lambda x: bool(distutils.util.strtobool(x)), default=False)
    parser.add_argument('--show-loss-max', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
    parser.add_argument('--loss-max-num', type=int, default=50)
    return parser.parse_args()


def main():
    args = arg_parse()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    '''new'''
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
    else:
        pass

    C.merge_from_file(args.cfg)
    C.DATA_ROOT = utils.get_data_root()
    C.freeze()
    model_name = args.ckpt.split('/')[-2]
    iter_name = args.ckpt.split('/')[-1].split('.')[0]
    output_dir = os.path.join('eval_vis', model_name, iter_name+'-'+args.data_dir.replace('/','-'))

    dataset = eval(f'{C.DATASET_ABS}')(data_dir=os.path.join(C.DATA_ROOT, args.data_dir), phase='test', pred_all=args.pred_all)
    data_loader = DataLoader(dataset, batch_size=C.SOLVER.VAL_BATCH_SIZE, num_workers=0)
    print("(loss)pred step:", dataset.pred_steps)

    model = FGNS()
    if torch.cuda.is_available():
        device = torch.device(C.DEVICE)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    model.to(device)

    if torch.cuda.is_available():
        cp = torch.load(args.ckpt, map_location=C.DEVICE)
    else:
        cp = torch.load(args.ckpt, map_location=f'cpu')

    model.load_state_dict(cp['model'])

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter:",  total)
    print("Number of parameter: %.2fM" % (total / 1e6))

    exp_name_list = args.ckpt.split('/')
    exp_name = exp_name_list[1] + '-' + exp_name_list[2].split('_')[-1].split('.')[0]

    tester = PredEvaluator(
        device=device,
        data_loader=data_loader,
        model=model,
        output_dir=output_dir,
        exp_name=exp_name,
        output_gif=args.output_gif,
        if_sort=args.show_loss_max,
        loss_max_num=args.loss_max_num,
        dataset=args.dataset_name,
        pred_all=args.pred_all,
    )
    tester.test()


if __name__ == '__main__':
    main()

    # python eval.py --cfg configs/slide_plus.yaml --ckpt ckpts/slide_plus-test/best_*****.path.tar --data-dir Slide_Plus/test
