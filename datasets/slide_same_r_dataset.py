import os
import time

import torch
import random
from random import sample
from torch.utils.data import Dataset
from config import _C as C
import utils
import pickle
import json
import numpy as np
import h5py
import itertools
from glob import glob
from utils import tprint, ture_rollout_index
from config import _C as C
import torch.nn.functional as F


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


class SlideSameRDataset(Dataset):
    def __init__(self, data_dir, phase='train', pred_all=True):
        self.data_dir = data_dir
        self.phase = phase
        self.metadata = _read_metadata(os.path.abspath(os.path.join(data_dir, "..")))

        for key in self.metadata:
            self.metadata[key] = torch.from_numpy(np.array(self.metadata[key]).astype(np.float32))

        self.rollout_steps = C.ROLLOUT_STEPS

        if self.phase == 'val':
            self.pred_steps = C.ROLLOUT_STEPS - C.N_HIS
        elif self.phase == 'test':
            if pred_all:
                self.pred_steps = C.ROLLOUT_STEPS - C.N_HIS
            else:
                self.pred_steps = C.PRED_STEPS
        else:
            self.pred_steps = C.PRED_STEPS
        self.length = self.get_len()

    def get_len(self):
        all_num_vids = len(glob(os.path.join(self.data_dir, '*')))

        if self.phase == 'val':
            num_vids = min(all_num_vids, C.MAX_VAL)
        elif self.phase == 'test':
            num_vids = min(all_num_vids, C.MAX_TEST)
        elif self.phase == "train":
            num_vids = min(all_num_vids, C.MAX_TRAIN)
        else:
            num_vids = all_num_vids

        if self.phase == "train":
            self.time_step_per_file = C.ROLLOUT_STEPS - C.N_HIS - self.pred_steps + 1
        else:
            self.time_step_per_file = C.ROLLOUT_STEPS - C.N_HIS - self.pred_steps + 1

        num = self.time_step_per_file * num_vids

        return num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        idx_rollout = idx // self.time_step_per_file
        idx_timestep = (C.N_HIS - 1) + (idx % self.time_step_per_file) * C.TRAIN_INTERVAL

        self.idx_timestep = idx_timestep

        data = self._load_data_file(idx_rollout)

        poss = data['position'][:, idx_timestep - C.N_HIS + 1:idx_timestep + 1]
        tgt_poss = data['position'][:, idx_timestep + 1:idx_timestep + self.pred_steps + 1]

        nonk_mask = get_non_kinematic_mask(data['particle_type'])

        if self.phase == 'train':
            sampled_noise = utils.get_random_walk_noise(poss, idx_timestep, C.NET.NOISE)
            sampled_noise = sampled_noise * nonk_mask[:, None, None]
            poss = poss + sampled_noise

            tgt_poss = tgt_poss + sampled_noise[:, -1:]

        vels = utils.time_diff(np.concatenate([poss, tgt_poss], axis=1))
        accs = utils.time_diff(vels)

        tgt_vels = vels[:, -self.pred_steps:]
        tgt_accs = accs[:, -self.pred_steps:]
        vels = vels[:, :-self.pred_steps]

        poss = torch.from_numpy(poss.astype(np.float32))
        vels = torch.from_numpy(vels.astype(np.float32))
        tgt_vels = torch.from_numpy(tgt_vels.astype(np.float32))
        tgt_accs = torch.from_numpy(tgt_accs.astype(np.float32))
        particle_type = torch.from_numpy(data['particle_type']).int()
        particle_r = torch.from_numpy(data['particle_r'].astype(np.float32))
        particle_m = torch.from_numpy(data['particle_m'].astype(np.float32))
        nonk_mask = torch.from_numpy(nonk_mask.astype(np.int32))
        tgt_poss = torch.from_numpy(tgt_poss.astype(np.float32))

        return poss, vels, tgt_accs, tgt_vels, particle_r / C.NET.RADIUS, particle_type, nonk_mask, tgt_poss, idx, particle_m

    def _load_data_file(self, idx_rollout):
        file = os.path.join(self.data_dir, f'{idx_rollout}.pkl')
        data = pickle.load(open(file, 'rb'))
        data['position'] = data['position'].transpose([1, 0, 2])
        return data


def get_non_kinematic_mask(particle_types):
    """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
    return particle_types != C.KINEMATIC_PARTICLE_ID
