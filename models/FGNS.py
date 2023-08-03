import copy

import numpy as np
import torch
import torch.nn as nn

import utils
from models.layers.layer import GNN
from scipy import spatial
from config import _C as C
from utils import decompose


class FGNS(nn.Module):
    def __init__(self):
        super(FGNS, self).__init__()
        self.hidden_size = C.NET.HIDDEN_SIZE
        self.R = C.NET.RADIUS
        self.particle_type_num = C.NUM_PARTICLE_TYPES
        self.particle_type_emb_dim = C.NET.PARTICLE_TYPE_EMB_SIZE
        self.edge_type_emb_dim = C.NET.EDGE_TYPE_EMB_SIZE
        self.edge_type_num = (self.particle_type_num - 1) * self.particle_type_num / 2 + self.particle_type_num
        self.direction_type_num = 14
        self.direction_type_emb_dim = C.NET.DIRECTION_TYPE_EMB_SIZE

        self.particle_type_emb = nn.Embedding(self.particle_type_num, self.particle_type_emb_dim)
        self.edge_emb = nn.Embedding(int(self.edge_type_num), self.edge_type_emb_dim)
        self.edge_direction_emb = nn.Embedding(self.direction_type_num, self.direction_type_emb_dim)

        self.node_dim_in = (C.N_HIS - 1) * (2 + 1) + (C.N_HIS - 2) + + self.particle_type_emb_dim + 4 + 1 + 1
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.edge_dim_in = 3 + 3 + self.edge_type_emb_dim + self.direction_type_emb_dim
        self.normal_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in + 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.tangential_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in + 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.damping_edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim_in, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.graph = GNN(layers=C.NET.GNN_LAYER)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

    def construct_edge(self, poss, nonk_mask):
        device = poss.device
        collapsed = False

        n_particles = poss.shape[0]
        # Calculate undirected edge list using KDTree
        point_tree = spatial.cKDTree(poss.detach().cpu().numpy())
        undirected_pairs = np.array(list(point_tree.query_pairs(C.NET.RADIUS, p=2))).T
        undirected_pairs = torch.from_numpy(undirected_pairs).to(device)
        pairs = torch.cat([undirected_pairs, torch.flip(undirected_pairs, dims=(0,))], dim=1).long()

        if C.NET.SELF_EDGE:
            self_pairs = torch.stack([torch.arange(n_particles, device=device),
                                    torch.arange(n_particles, device=device)])
            pairs = torch.cat([pairs, self_pairs], dim=1)

        senders = pairs[0]
        receivers = pairs[1]

        senders_mask = nonk_mask[senders]
        receivers_mask = nonk_mask[receivers]
        both_nonk_mask = ((senders_mask + receivers_mask) != 0)

        senders = senders[both_nonk_mask]
        receivers = receivers[both_nonk_mask]

        if pairs.shape[1] > C.NET.MAX_EDGE_PER_PARTICLE * n_particles:
            collapsed = True

        return senders, receivers, collapsed

    def node_constructor(self, v, metadata):
        time_step = v.shape[1]
        last_step_scalar_v_x, direct_v_x = decompose(v[:, -1, 0])
        other_scalar_v_x = torch.abs(v[:, :-1, 0])
        scalar_v_x = torch.cat([other_scalar_v_x, last_step_scalar_v_x.reshape(-1, 1)], dim=1)

        scalar_v_x = (scalar_v_x - metadata['scalar_vel_mean'][0]) / metadata['scalar_vel_std'][0]
        vector_v_y = (v[:, :, 1] - metadata['vel_mean'][1]) / metadata['vel_std'][1]
        vels = torch.cat([scalar_v_x.reshape(-1, time_step), vector_v_y.reshape(-1, time_step)], dim=1)

        # Calculate particle velocity ｜v｜
        v_value = torch.norm(v, dim=2)
        v_value = (v_value - metadata['V_mean']) / metadata['V_std']
        v_value = v_value.reshape(-1, time_step)

        dir_v_x_change = self.direct_v_x_change(v[:, :, 0])  # The length is 0 and currently has no effect

        # particle type t
        type_emb = self.particle_type_emb(self.particle_type)

        # Calculate the distance from the wall
        last_step_poss = self.poss[:, -1]
        dist_to_walls = torch.cat(
            [last_step_poss - metadata['out_bounds'][:, 0],
             -last_step_poss + metadata['out_bounds'][:, 1]], 1)
        dist_to_walls = torch.clip(dist_to_walls / C.NET.RADIUS, -1, 1)
        dist_to_walls_x = torch.cat([dist_to_walls[:, 0].reshape(-1, 1), dist_to_walls[:, 2].reshape(-1, 1)], 1)
        dist_to_walls_x_op = torch.cat([dist_to_walls[:, 2].reshape(-1, 1), dist_to_walls[:, 0].reshape(-1, 1)], 1)
        dist_to_walls_y = torch.cat([dist_to_walls[:, 1].reshape(-1, 1), dist_to_walls[:, 3].reshape(-1, 1)], 1)

        dist_to_walls_x = self.change_dist_to_walls(dist_to_walls_x, dist_to_walls_x_op, direct_v_x)
        dist_to_walls = torch.cat([dist_to_walls_x, dist_to_walls_y], dim=1)

        # particle radius r
        particle_r = self.particle_r.reshape(-1, 1)
        particle_r_2 = particle_r ** 2

        input = torch.cat([vels, v_value, dir_v_x_change, type_emb, dist_to_walls, particle_r, particle_r_2], dim=1)

        node = self.node_encoder(input)

        return node, direct_v_x, v

    def direct_v_x_change(self, v_x):
        plus = v_x > 0
        minus = v_x < 0
        zeros = (v_x == 0)
        direction = plus * 1 + minus * -1 + zeros * 0

        direct_v = direction[torch.arange(direction.size(0)), (direction != 0).long().sum(dim=1).sub(1)]
        direct_v[direct_v == 0] = 1

        direction_diff = utils.time_diff(direction) * direct_v.unsqueeze(1)

        return direction_diff

    def change_dist_to_walls(self, dist_to_walls, dist_to_walls_op, direct_v):
        in_boundary = (dist_to_walls[:, 0] > 0) * (dist_to_walls[:, 1] > 0)
        out_boundary = (in_boundary != 1)
        plus = direct_v > 0
        zero = direct_v == 0
        miuns = direct_v < 0

        change = torch.abs(dist_to_walls[:, 0]) > torch.abs(dist_to_walls[:, 1])

        dist_to_walls[plus * in_boundary] = dist_to_walls_op[plus * in_boundary]
        need_change = ((zero + out_boundary) > 0) * change

        dist_to_walls[need_change] = dist_to_walls_op[need_change]

        return dist_to_walls

    def edge_constructor(self, direct_v_x):
        last_step_poss = self.poss[:, -1]
        dist_vec = (last_step_poss[self.receivers] - last_step_poss[self.senders])
        dist_vec = dist_vec / self.R

        non_self_edge_num = self.edge_num - self.kinematic_num

        v = self.vels[:, -1]
        # Decomposition vel direction
        normal_senders_cos, tangential_senders_sin = utils.vel_decomposition(
            v[self.senders[:non_self_edge_num]], dist_vec[:non_self_edge_num])
        normal_receivers_cos, tangential_receivers_sin = utils.vel_decomposition(
            v[self.receivers[:non_self_edge_num]], dist_vec[:non_self_edge_num])

        dist = torch.linalg.norm(dist_vec, dim=1, keepdims=True).reshape(-1)
        dist_vec, direct_dist_x = utils.xy_decompose(dist_vec)

        senders_r = self.particle_r[self.senders]
        receivers_r = self.particle_r[self.receivers]
        h = dist - senders_r - receivers_r
        theta = h / dist
        theta[torch.isnan(theta)] = 0
        theta[torch.isinf(theta)] = 0
        dist_without_r_vec = theta.unsqueeze(1) * dist_vec

        # edge type T
        senders_type = self.particle_type[self.senders]
        receivers_type = self.particle_type[self.receivers]
        senders_type, receivers_type = self.change_r_s(senders_type, receivers_type)
        edge_type = (receivers_type + 1) * receivers_type / 2 + senders_type
        edge_type_emb = self.edge_emb(edge_type.long())

        normal_msg = torch.cat([normal_senders_cos.reshape(-1, 1), normal_receivers_cos.reshape(-1, 1)], dim=1)

        tangential_msg = torch.cat([tangential_senders_sin.reshape(-1, 1), tangential_receivers_sin.reshape(-1, 1)], dim=1)

        # generate sym direction encoding
        direction_encoding = self.get_direction_encoding(direct_dist_x, direct_v_x)

        dist = dist.reshape(-1, 1)
        h = h.reshape(-1, 1)
        input = torch.cat([dist_vec, dist, dist_without_r_vec, h, edge_type_emb, direction_encoding], dim=1)

        # Generate three directional features
        normal_edge = self.normal_edge_encoder(torch.cat([input[:non_self_edge_num], normal_msg], dim=1))
        tangential_edge = self.tangential_edge_encoder(torch.cat([input[:non_self_edge_num], tangential_msg], dim=1))
        damping_edge = self.damping_edge_encoder(input[non_self_edge_num:])

        return normal_edge, tangential_edge, damping_edge

    def get_direction_encoding(self, direct_dist, direct_v):
        direct_v_s = direct_v[self.senders]
        direct_v_r = direct_v[self.receivers]

        direct_dist += 1
        direct_v_s += 1
        direct_v_r += 1

        direction_type = torch.abs((self.direction_type_num - 1) - (direct_dist * 9 + direct_v_s * 3 + direct_v_r * 1))
        direction_encoding = self.edge_direction_emb(direction_type)

        return direction_encoding

    def change_r_s(self, s, r):
        mask = r < s
        adjusted_r = torch.where(mask, s, r)
        adjusted_s = torch.where(mask, r, s)
        return adjusted_s, adjusted_r

    def waste_direction(self, acc_waste, direct_v):
        zeros = (direct_v == 0)
        direct_v[zeros] = 1
        return acc_waste * direct_v

    def forward(self, poss, vels, particle_r, particle_type, nonk_mask, metadata, tgt_poss, tgt_vels, num_rollouts):
        self.particle_type = particle_type
        self.particle_r = particle_r
        self.nonk_mask = nonk_mask
        self.node_num = poss.shape[0]
        self.kinematic_num = torch.sum(nonk_mask)
        self.nonk_num = self.node_num - self.kinematic_num
        self.poss = poss
        self.vels = vels

        pred_accns = []
        pred_vels_list = []
        pred_poss_list = []

        for i in range(num_rollouts):
            self.senders, self.receivers, collapsed = self.construct_edge(self.poss[:, -1], nonk_mask)

            self.edge_num = self.senders.shape[0]

            node, direct_v_x, v = self.node_constructor(v=self.vels, metadata=metadata)

            normal_edge, tangential_edge, damping_edge = self.edge_constructor(direct_v_x)

            non_self_edge_num = self.edge_num - self.kinematic_num
            node = self.graph(
                node, normal_edge, tangential_edge, damping_edge,
                self.senders[:non_self_edge_num], self.receivers[:non_self_edge_num],
                self.senders[non_self_edge_num:], self.receivers[non_self_edge_num:])

            pred_accn = self.decoder(node)

            pred_accn_x = pred_accn[:, 0]

            pred_accn_x = self.waste_direction(pred_accn_x, direct_v_x)
            pred_accn[:, 0] = pred_accn_x

            pred_acc = pred_accn * metadata['acc_std'] + metadata['acc_mean']

            pred_accns.append(pred_accn)

            pred_vels = self.vels[:, -1] + pred_acc
            pred_poss = self.poss[:, -1] + pred_vels

            pred_vels = torch.where(nonk_mask[:, None].bool(), pred_vels, tgt_vels[:, i])
            pred_poss = torch.where(nonk_mask[:, None].bool(), pred_poss,
                                    tgt_poss[:, i])
            pred_vels_list.append(pred_vels)
            pred_poss_list.append(pred_poss)

            if i < num_rollouts - 1:
                self.poss = torch.cat([self.poss[:, 1:], pred_poss[:, None]], dim=1)
                self.vels = torch.cat([self.vels[:, 1:], pred_vels[:, None]], dim=1)

            if collapsed:
                break

        pred_accns = torch.stack(pred_accns).permute(1, 0, 2)
        pred_poss = torch.stack(pred_poss_list).permute(1, 0, 2)
        pred_vels = torch.stack(pred_vels_list).permute(1, 0, 2)

        outputs = {
            'pred_accns': pred_accns,
            'pred_vels': pred_vels,
            'pred_poss': pred_poss,
            'pred_collaposed': collapsed,
        }

        return outputs





