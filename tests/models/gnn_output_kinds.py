"""GNN fixtures whose outputs are edge-level and fixed-size, next to gnn_gcn's node-level one."""

import torch
from torch import nn
from torch_geometric.data import Data


class EdgeMLP(nn.Module):
    """Scores each edge from the concatenated features of its two endpoints: [E, 1]."""

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * in_channels, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, data: Data) -> torch.Tensor:
        src, dst = data.edge_index
        return self.mlp(torch.cat([data.x[src], data.x[dst]], dim=-1))


class SumReadout(nn.Module):
    """Sums the node features into one row: [1, F], whatever the graph size."""

    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        return self.lin(data.x.sum(0, keepdim=True))


def make_edge_model() -> EdgeMLP:
    return EdgeMLP().eval()


def make_fixed_model() -> SumReadout:
    return SumReadout().eval()


def make_inputs(num_nodes: int = 6, num_edges: int = 10) -> tuple[Data]:
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return (Data(x=x, edge_index=edge_index),)
