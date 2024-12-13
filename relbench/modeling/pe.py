from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from relbench.modeling.gin import GIN
from relbench.modeling.mlp import MLP

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from scipy.special import comb
import math

def lap_filter(L: torch.Tensor, W: torch.Tensor, k: int) -> torch.Tensor:
    '''
    This function represents a graph filter that repeatedly applies the graph Laplacian
    to a matrix of node features, returning [W, LW, L^2W, ..., L^kW]
    
    Args:
        L: Laplacian matrix, shape [N, N].
        W: Node feature matrix, shape [N,N] w/ basis vectors and [N, M] w/ random samples.
        k: The order (k) of the Laplacian filter.   
        returns: [N, M, k].
    '''
    output = W
    w_list = [output.unsqueeze(-1)]
    for _ in range(k-1):
        output = L @ output
        w_list.append(output.unsqueeze(-1))
    return torch.cat(w_list, dim=-1)


class K_PEARL_PE(nn.Module):
    '''
        Structure adapted from https://github.com/Graph-COM/SPE.git 
        This is a part of the PEARL positional encoder. This model takes in random or
        basis vectors as input features, applies the laplacian filter and passes the output
        through an MLP before passing it through phi. 

        Parameters:
            phi: A model that 

    '''
    def __init__(
        self, 
        phi: nn.Module, 
        k: int = 16, 
        mlp_nlayers: int = 1, 
        mlp_hid: int = 16, 
        spe_act: str = 'relu', 
        mlp_out: int = 16
    ) -> None:
        if mlp_nlayers > 0:
            self.bn = nn.ModuleList()
            self.layers = nn.ModuleList([
                nn.Linear(
                    k if i==0 else mlp_hid, mlp_hid if i<mlp_nlayers-1 else mlp_out, bias=True) 
                    for i in range(mlp_nlayers)
                ])
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(
                    mlp_hid if i<mlp_nlayers-1 else mlp_out,track_running_stats=True) 
                    for i in range(mlp_nlayers)
                ])
        if spe_act == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif spe_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        self.running_sum = 0
        self.total = 0

    def forward(
        self, Lap, W, edge_index: torch.Tensor, final=False
    ) -> torch.Tensor:
        """
        Parameters:
            Lap: NxN Laplacian
            W: the batched B*[NxM] or BxNxN list of basis vectors (B is usually 1)
            edge_index: Graph connectivity in COO format. [2, E_sum]
            batch: Batch index vector. [N_sum]
            return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        # Here we loop through each graph and its laplacian, but our batch size is usually 1 subgraph.
        for lap, w in zip(Lap, W):
            output = lap_filter(lap, w, self.k)
            if self.mlp_nlayers > 0:
                for layer, bn in zip(self.layers, self.norms):
                    # Reshape [N, M, K] to [M, N, K] to ensure the linear layer processes each M samples independently.
                    output = output.transpose(0, 1)
                    output = layer(output)
                    # Reshape [M,N,K] to [M,K,N] before applying batchnorm to normalize across the K-th 
                    # dimension and preserve sample or basis independence
                    output = bn(output.transpose(1, 2)).transpose(1, 2)
                    output = self.activation(output)
                    output = output.transpose(0, 1) # Transpose back to original [N,M,K shape]
            W_list.append(output) 
        return self.phi(W_list, edge_index, final=final)   # This returns shape [N, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims


class SignInvPe(nn.Module):
    '''
        SignNet model from https://github.com/Graph-COM/SPE.git 
    '''
    def __init__(self, phi: nn.Module, rho: nn.Module) -> None:
        super(SignInvPe, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(
            self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = V.unsqueeze(-1) 
        x = self.phi(x, edge_index) + self.phi(-x, edge_index) # [N, D_pe, hidden_dims]
        x = x.reshape([x.shape[0], -1]) # [N, D_pe * hidden_dims]
        x = self.rho(x) # [N, D_pe]

        return x

    @property
    def out_dims(self) -> int:
        return self.rho.out_dims


class GINPhi(nn.Module):
    '''
        Framework adapted from https://github.com/Graph-COM/SPE.git
        This model uses a special GIN that is meant to take in a 3d tensor of shape 
        [N,M,K] or [N,N,K] and processes the second dimension independently, preserving
        independence across the M random vectors or N basis vectors. 
    '''
    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool
    ) -> None:
        super().__init__()
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn)
        self.running_sum = 0
        self.total = 0

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, final: bool =False) -> torch.Tensor:
        """
        Args:
            W_list: The list of NxMxK (or NxNxK if using basis vectors) tensors per graph in the batch, 
                has shape [N, M, K] * B.
            edge_index: Graph connectivity in COO format. [2, E_sum]
            final: Whether or not we are finished passing in our samples or basis vectors and need to reset the running sum.
            return: Positional encoding matrix. [N_sum, D_pe]
        Here N_sum refers to the number of nodes aggregated across graphs in the batch. However,
        since we sample one subgraph at a time (B=1), N_sum=N
        """ 
        # If we are using random samples,
        W = torch.cat(W_list, dim=0)   # [N_sum, M, K]
        PE = self.gin(W, edge_index)  # [N,M,D]
        self.running_sum += (PE).sum(dim=1) # Keep track of the running sum
        if final:
            PE = self.running_sum
            self.running_sum = 0
        return PE               # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


'''
    Helper function to create GINPhi model.
'''
def GetPhi(cfg, create_mlp: Callable[[int, int], MLP], device):
    return GINPhi(cfg.n_phi_layers, cfg.RAND_mlp_out, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm, RAND_LAP=cfg.RAND_LAP)