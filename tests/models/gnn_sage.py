"""GNN fixture: 2-layer GraphSAGE node classifier."""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_channels: int = 8, hidden: int = 16, out_channels: int = 4):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def make_model() -> SAGE:
    model = SAGE()
    model.eval()
    return model


def make_inputs(num_nodes: int = 6, num_edges: int = 10) -> tuple[Data]:
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (Data(x=x, edge_index=edge_index),)
