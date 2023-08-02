import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import _C as C


class MessagePassing(nn.Module):
    def __init__(self, layer):
        super(MessagePassing, self).__init__()
        self.hidden_size = C.NET.HIDDEN_SIZE

        self.normal_edge_model = nn.Sequential(
            nn.Linear(self.hidden_size*3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.tangential_edge_model = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.damping_edge_model = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.node_model = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

    def forward(self, node, normal_edge, tangential_edge, damping_edge, senders, receivers, self_edge_senders, self_edge_receivers):
        senders_node = node[senders]
        receivers_node = node[receivers]

        out_normal_edge = self.normal_edge_model(torch.cat([senders_node, receivers_node, normal_edge], dim=1))
        normal_effect = torch.zeros_like(node).scatter_add_(0, receivers[:, None].expand(out_normal_edge.shape), out_normal_edge)

        out_tangential_edge = self.tangential_edge_model(torch.cat([senders_node, receivers_node, tangential_edge], dim=1))
        tangential_effect = torch.zeros_like(node).scatter_add_(0, receivers[:, None].expand(out_tangential_edge.shape), out_tangential_edge)

        self_edge_senders_node = node[self_edge_senders]
        self_edge_receivers_node = node[self_edge_receivers]

        out_damping_edge = self.damping_edge_model(torch.cat([self_edge_senders_node, self_edge_receivers_node, damping_edge], dim=1))
        damping_effect = torch.zeros_like(node).scatter_add_(0, self_edge_receivers[:, None].expand(out_damping_edge.shape), out_damping_edge)

        effect = normal_effect + tangential_effect + damping_effect
        out_node = self.node_model(torch.cat([effect, node], dim=1))

        return out_node, out_normal_edge, out_tangential_edge, out_damping_edge


class GNN(nn.Module):
    def __init__(self, layers=1):
        super(GNN, self).__init__()
        self.layers = layers
        self.gn_list = nn.ModuleList([MessagePassing(i) for i in range(self.layers)])

    def forward(self, node, normal_edge, tangential_edge, damping_edge, senders, receivers, self_edge_senders, self_edge_receivers):
        for i, l in enumerate(self.gn_list):

            out_node, out_normal_edge, out_tangential_edge, out_damping_edge = l(node, normal_edge, tangential_edge, damping_edge, senders, receivers, self_edge_senders, self_edge_receivers)

            node += out_node
            normal_edge += out_normal_edge
            tangential_edge += out_tangential_edge
            damping_edge += out_damping_edge

        return node
