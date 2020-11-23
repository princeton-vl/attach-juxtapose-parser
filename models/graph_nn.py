"""
GNNs used for encoding the partial tree
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import List


class SeparateGCNConv(MessagePassing):  # type: ignore
    """
    A variant of GCN that separates content and position information
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    """

    d_content: int
    linear_c: nn.Module
    linear_p: nn.Module

    def __init__(self, d_model: int) -> None:
        super().__init__(aggr="add")
        self.d_content = d_model // 2
        d_positional = d_model - self.d_content
        self.linear_c = nn.Linear(self.d_content, self.d_content)
        self.linear_p = nn.Linear(d_positional, d_positional)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # apply linear transformations separately to content and position
        x_c = self.linear_c(x[:, : self.d_content])
        x_p = self.linear_p(x[:, self.d_content :])
        x = torch.cat([x_c, x_p], dim=-1)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y: torch.Tensor = self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm
        )
        return y

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


class GraphNeuralNetwork(nn.Module):
    """
    Multiple GCN layers with layer normalization and residual connections
    """

    convs: List[nn.Module]

    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()  # type: ignore
        self.convs = []
        self.layernorms = []
        for i in range(num_layers):
            conv = SeparateGCNConv(d_model)
            layernorm = nn.LayerNorm([d_model])
            self.convs.append(conv)
            self.layernorms.append(layernorm)
            self.add_module("conv_%d" % i, conv)
            self.add_module("layernorm_%d" % i, layernorm)

    def forward(
        self, graph: torch_geometric.data.Batch, nodes_of_interest: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index

        for conv, layernorm in zip(self.convs[:-1], self.layernorms[:-1]):
            x = x + F.relu(layernorm(conv(x, edge_index)))

        x = x + self.layernorms[-1](self.convs[-1](x, edge_index))

        return x[nodes_of_interest]  # type: ignore
