from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj , degree
from relbench.modeling.mlp import MLP

class GIN(nn.Module):
    '''
        GIN module from https://github.com/Graph-COM/SPE.git 
        This special GIN model takes in a 3D tensor of shape [N,M,K], and preserves sample
        independence across the M samples, processing each sample indepenently.
    '''

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            new_mlp = create_mlp(in_dims, hidden_dims)
            layer = GINLayer(new_mlp)
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
        new_m = create_mlp(hidden_dims, out_dims)
        layer = GINLayer(new_m)
        self.layers.append(layer)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index, mask=mask)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            if mask is not None:
                X[~mask] = 0
            if self.bn and i < len(self.layers) - 1:
                if mask is None:
                    if X.ndim == 3:
                        X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                    else:
                        X = self.batch_norms[i](X)
                else:
                    X[mask] = self.batch_norms[i](X[mask])
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    '''
        Simple GIN layer from https://github.com/Graph-COM/SPE.git 
    '''

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True) #torch.empty(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """

        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z, mask)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims