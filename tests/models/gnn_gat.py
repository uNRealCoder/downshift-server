"""GNN fixture: 3-layer GAT node classifier.

GATConv's attention-weighted aggregation is the scatter-heavy pattern most likely to trip
the ONNX translation of scatter_reduce. On torch 2.14 / torch_geometric 2.8 it exports
CLEAN; see scatter_include_self_false for the case that doesn't.
"""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels: int = 8, hidden: int = 8, out_channels: int = 4, heads: int = 2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads)
        self.conv3 = GATConv(hidden * heads, out_channels, heads=1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)


def make_model() -> GAT:
    model = GAT()
    model.eval()
    return model


def make_inputs(num_nodes: int = 6, num_edges: int = 10) -> tuple[Data]:
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (Data(x=x, edge_index=edge_index),)
