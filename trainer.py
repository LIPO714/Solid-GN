import os
import time

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import gradcheck

from utils import tprint
from timeit import default_timer as timer
from config import _C as C
from torch.utils.tensorboard import SummaryWriter
import utils


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim,
                 max_iters, exp_name, tired_iters, dataset):
        self.device = device
        self.exp_name = exp_name

        self.train_loader, self.val_loader = train_loader, val_loader
        self.metadata = utils.update_metadata(self.train_loader.dataset.metadata, self.device)  # 读取元数据，并对速度和加速度的标准值增加噪声

        self.model = model
        self.optim = optim
        self.iterations = 0
        self.tired_now = 0
        self.tired = 0
        self.tired_iters = tired_iters
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        self.grad_accumulation_steps = C.SOLVER.GRAD_ACCUMULATION_STEP
        self.dataset = dataset
        self.train_loader_length = len(train_loader)

        self._setup_loss()
        self.setup_dirs()

        self.tb_writer = SummaryWriter(self.log_dir)

    def setup_dirs(self):
        self.log_dir = f'./logs/{self.exp_name}'
        self.ckpt_dir = f'./ckpts/{self.exp_name}'

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        print_msg = "| ".join(["epoch ", "batch iter "] + list(map("{:6}".format, self.loss_name)))
        self.model.train()
        print('\r', end='')
        print(print_msg)
        self.best_val_pos_loss = 1e7
        while self.iterations < self.max_iters and not self.tired:
            if self.iterations >= C.SOLVER.FAST_VAL_EPOCH:
                self.val_interval = C.SOLVER.FAST_VAL_INTERVAL
            self.train_epoch()

            self.iterations += 1

    def train_epoch(self):
        batch_iter = 0
        self.optim.zero_grad()
        for batch_idx, data in enumerate(self.train_loader):
            self.tt = timer()

            assert data[0].shape[0] == 1

            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            poss, vels, tgt_accs, tgt_vels, particle_r, particle_type, nonk_mask, tgt_poss, idx, particle_m = data

            num_rollouts = tgt_vels.shape[1]  # 1
            outputs = self.model(poss, vels, particle_r, particle_type, nonk_mask, self.metadata, tgt_poss, tgt_vels,
                                 num_rollouts=num_rollouts)
            # poss_overturn, vels_overturn, particle_type_overturn, nonk_mask_overturn, tgt_poss_overturn, tgt_vels_overturn = self.overturn(poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels, num_rollouts=num_rollouts)
            # outputs_overturn = self.model(poss_overturn, vels_overturn, particle_r, particle_type_overturn, nonk_mask_overturn, self.metadata, tgt_poss_overturn, tgt_vels_overturn, num_rollouts=num_rollouts)

            # poss_move, vels_move, particle_type_move, nonk_mask_move, tgt_poss_move, tgt_vels_move, metadata_move = self.move(10, poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels)
            # outputs_move = self.model(poss_move, vels_move, particle_r, particle_type_move, nonk_mask_move, metadata_move, tgt_poss_move, tgt_vels_move, num_rollouts=num_rollouts)

            tgt_accns = (tgt_accs - self.metadata['acc_mean']) / self.metadata['acc_std']  # 标准化 使其符合正态分布

            labels = {
                'accns': tgt_accns,
                'poss': tgt_poss,
            }

            weight = nonk_mask

            loss = self.loss(outputs, labels, weight, 'train')  # 计算loss
            loss = loss / self.grad_accumulation_steps
            loss.backward()

            print_msg = ""
            print_msg += f"{self.iterations}"
            print_msg += f" | "
            print_msg += f"{batch_iter}"
            print_msg += f" | "
            print_msg += f" | ".join(
                ["{}:{:.12f}".format(name, self.single_losses[name]) for name in self.loss_name])  # 当前这个batch的loss
            step_time = timer() - self.tt
            self.tt = timer()
            eta = (
                          self.max_iters * self.train_loader_length - self.iterations * self.train_loader_length + batch_iter) * step_time / 3600
            print_msg += f" | step time: {step_time:.2f} | eta: {eta:.2f} h"
            tprint(print_msg)

            for name in self.loss_name:
                self.tb_writer.add_scalar(f'Single/{name}', self.single_losses[name],
                                          self.iterations * self.train_loader_length + batch_iter)  # 当前batch的loss
                self.tb_writer.add_scalar(f'Period/{name}', self.period_losses[name] / self.loss_cnt,
                                          self.iterations * self.train_loader_length + batch_iter)  # 从上一次val开始，截止到现在各个batch的平均值

            if ((batch_iter + 1) % self.grad_accumulation_steps == 0) or batch_iter == len(self.train_loader) - 1:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 设置裁剪的最大范数
                is_nan = False
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("Warning: NaN gradients detected. Skipping batch.")
                        is_nan = True
                        break
                if not is_nan:
                    self.optim.step()
                    # print("optim step")

                self.optim.zero_grad()
                self._adjust_learning_rate()

            if (batch_iter + self.iterations * self.train_loader_length) % self.val_interval == 0 and (
                    self.iterations != 0 or batch_iter != 0):
                self.snapshot(batch_iter + self.iterations * self.train_loader_length)
                self.val(batch_iter)
                if self.tired == 1:
                    break
                self._init_loss()
                self.model.train()

            batch_iter += 1

    def val(self, batch_iter):
        self.model.eval()
        self._init_loss()

        for batch_idx, data in enumerate(self.val_loader):
            assert data[0].shape[0] == 1

            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            poss, vels, tgt_accs, tgt_vels, particle_r, particle_type, nonk_mask, tgt_poss, _, _ = data

            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')

            with torch.no_grad():

                num_rollouts = tgt_vels.shape[1]

                outputs = self.model(poss, vels, particle_r, particle_type, nonk_mask, self.metadata, tgt_poss,
                                     tgt_vels, num_rollouts=num_rollouts)

                tgt_accns = (tgt_accs - self.metadata['acc_mean']) / self.metadata['acc_std']

                labels = {
                    'accns': tgt_accns,
                    'poss': tgt_poss,
                }

                self.loss(outputs, labels, nonk_mask, 'test')

                if outputs['pred_collaposed']:
                    break

        print("iterations:", self.iterations * self.train_loader_length + batch_iter, " ｜ epoch:", self.iterations,
              " | batch:", batch_iter)
        for name in self.loss_name:
            self.tb_writer.add_scalar(f'Val/{name}', self.period_losses[name] / self.loss_cnt,
                                      self.iterations * self.train_loader_length + batch_iter)
            print("losses[", name, "]:", self.period_losses[name] / self.loss_cnt)
            print('\n')

        if self.period_losses['pos'] < self.best_val_pos_loss:
            self.tired_now = 0
            self.best_val_pos_loss = self.period_losses['pos']
            self.snapshot_best(batch_iter + self.iterations * self.train_loader_length)
        else:
            self.tired_now += 1
            if self.tired_now >= self.tired_iters:
                print("Tired!!")
                self.tired = 1

    def loss(self, outputs, labels, weighting, phase):
        self.loss_cnt += 1  # ？

        if outputs['pred_collaposed']:
            print("pred collaposed")
            for name in self.loss_name:
                self.single_losses[name] = np.nan
                self.period_losses[name] = np.nan
            loss = np.nan
            return loss

        accn_loss = ((outputs['pred_accns'] - labels['accns']) * torch.unsqueeze(torch.unsqueeze(weighting, -1),
                                                                                 1)) ** 2
        accn_loss = accn_loss.mean(1).sum() / torch.sum(weighting)
        self.single_losses['accn'] = accn_loss.item()
        self.period_losses['accn'] += self.single_losses['accn']

        pos_loss = ((outputs['pred_poss'] - labels['poss']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        pos_loss = pos_loss.mean(1).sum() / torch.sum(weighting)
        self.single_losses['pos'] = pos_loss.item()
        self.period_losses['pos'] += self.single_losses['pos']

        if C.SOLVER.LOSS == "Acc":
            all_loss = accn_loss
        else:
            all_loss = pos_loss

        loss = all_loss
        self.single_losses['all'] = all_loss.item()
        self.period_losses['all'] += self.single_losses['all']

        print(loss.item())
        return loss

    def snapshot(self, iter):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.ckpt_dir, f'iter_{iter}.path.tar'),
        )

    def snapshot_best(self, iter):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.ckpt_dir, f'best_{iter}.path.tar'),
        )

    def _setup_loss(self):
        self.loss_name = ['accn', 'pos', 'all']
        self._init_loss()

    def _init_loss(self):
        self.single_losses = dict.fromkeys(self.loss_name, 0.0)
        self.period_losses = dict.fromkeys(self.loss_name, 0.0)
        self.loss_cnt = 0

    def _adjust_learning_rate(self):
        if self.iterations < C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS

        else:
            lr = (C.SOLVER.BASE_LR - C.SOLVER.MIN_LR) * 0.1 ** (
                        self.iterations / C.SOLVER.LR_DECAY_INTERVAL) + C.SOLVER.MIN_LR

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def overturn(self, poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels, num_rollouts):
        poss_overturn = poss
        poss_overturn[:,:,0] = self.metadata['bounds'][0][0] + self.metadata['bounds'][0][1] - poss_overturn[:,:,0]
        vels_overturn = vels
        vels_overturn[:,:,0] = -1 * vels_overturn[:,:,0]
        tgt_poss_overturn = tgt_poss
        tgt_poss_overturn[:,:,0] = self.metadata['bounds'][0][0] + self.metadata['bounds'][0][1] - tgt_poss_overturn[:,:,0]
        tgt_vels_overturn = tgt_vels
        tgt_vels_overturn[:,:,0] = -1 * tgt_vels_overturn[:,:, 0]
        return poss_overturn, vels_overturn, particle_type, nonk_mask, tgt_poss_overturn, tgt_vels_overturn

    def move(self, move_distance, poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels):
        poss[:,:,0] = poss[:,:,0] + move_distance
        tgt_poss[:,:,0] = tgt_poss[:,:,0] + move_distance
        metadata_move = self.metadata
        metadata_move['bounds'][0][0] += move_distance
        metadata_move['bounds'][0][1] += move_distance
        metadata_move['out_bounds'][0][0] += move_distance
        metadata_move['out_bounds'][0][1] += move_distance
        return poss, vels, particle_type, nonk_mask, tgt_poss, tgt_vels, metadata_move


