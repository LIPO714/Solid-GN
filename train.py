import distutils.util
import os
import torch
import random
import argparse
import numpy as np
from config import _C as cfg
from models.FGNS import FGNS
from trainer import Trainer
from datasets.slide_same_r_dataset import SlideSameRDataset
from datasets.slide_dataset import SlideDataset
from datasets.sand_dataset import SandDataset
from datasets.crash_dataset import CrashDataset
from datasets.slide_plus_dataset import SlidePlusDataset
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--cfg', required=True, help='config file', type=str)  # configs/slide.yaml
    parser.add_argument('--init', type=str, help='init model from', default='')
    parser.add_argument('--exp-name', type=str, help='exp name')  # slide-test
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--dataset-name', type=str, default="Slide", help='dataset name')  # Slide
    return parser.parse_args()


def main():
    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
    else:
        pass

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.DATA_ROOT = utils.get_data_root()
    cfg.freeze()

    # ---- setup model
    model = FGNS()
    # model.to(torch.device('cuda'))
    if torch.cuda.is_available():
        device = torch.device(cfg.DEVICE)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')

    print("device:", device)

    model.to(device)

    # if torch.cuda.is_available():
    #     # cp = torch.load('ckpts/best_30000_16_1_8.path.tar', map_location=cfg.DEVICE)
    #     cp = torch.load('ckpts/sand_2/best_1132000.path.tar', map_location=cfg.DEVICE)
    # # elif torch.backends.mps.is_available():
    # #     cp = torch.load(args.ckpt)
    # else:
    #     cp = torch.load(args.ckpt, map_location=f'cpu')
    #
    # model.load_state_dict(cp['model'])

    # for name, param in model.named_parameters():
    #     print(f'{name}: {param}')

    # ---- setup optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    if args.init:
        print(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    train_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.TRAIN_DIR), phase='train')
    val_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.VAL_DIR), phase='val')
    kwargs = {'pin_memory': False,
              'num_workers': 4
              }

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=cfg.SOLVER.TRAIN_SHUFFLE, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.SOLVER.VAL_BATCH_SIZE, shuffle=False, **kwargs,
    )

    print(f'epoch num:', cfg.SOLVER.MAX_ITERS)
    print(f'tired val num:', cfg.SOLVER.TIRED_ITERS)
    print(f'size: train {len(train_loader) * cfg.SOLVER.BATCH_SIZE} = {len(train_loader)} * {cfg.SOLVER.BATCH_SIZE} / val {len(val_loader) * cfg.SOLVER.VAL_BATCH_SIZE} = {len(val_loader)} * {cfg.SOLVER.VAL_BATCH_SIZE}')
    print(f'train batch size:{cfg.SOLVER.BATCH_SIZE} / val batch size:{cfg.SOLVER.VAL_BATCH_SIZE}')
    print(f'train interval:{cfg.TRAIN_INTERVAL}')

    kwargs = {'device': device,
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'exp_name': args.exp_name,
              'max_iters': cfg.SOLVER.MAX_ITERS,
              'tired_iters': cfg.SOLVER.TIRED_ITERS,
              'dataset': args.dataset_name,
              }
    trainer = Trainer(**kwargs)

    trainer.train()


if __name__ == '__main__':
    main()

    # --cfg configs/slide.yaml --exp-name slide-test --dataset-name Slide
    # --cfg configs/slide_same_r.yaml --exp-name slide_samer-test --dataset-name Slide_Same_R
    # --cfg configs/slide_plus.yaml --exp-name slide_plus-test --dataset-name Slide_Plus
    # --cfg configs/crash.yaml --exp-name crash-test --dataset-name Crash
    # --cfg configs/sand.yaml --exp-name sand-test --dataset-name Sand
