from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class DisenGCNLayer(MessagePassing, ABC):
    def __init__(self, message_dropout, temperature):
        super(DisenGCNLayer, self).__init__(aggr='add')
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(message_dropout)

    def forward(self, x, edge_index):
        c = self.dropout(x + self.propagate(edge_index, x=x))
        return c / torch.unsqueeze(torch.unsqueeze(torch.norm(c, 2, dim=[0, 2]), 0), 2)

    def message(self, x_i, x_j):
        p = torch.softmax(torch.matmul(x_i, x_j) / self.temperature, dim=0)
        return p * x_j