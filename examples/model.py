from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroGraphSAGE_LINK, HeteroEncoder_PEARL, HeteroGraphSAGE_PEARL, HeteroTemporalEncoder
from relbench.modeling.mlp import MLP as MLP_PE
from relbench.modeling.pe import K_PEARL_PE, GINPhi, SignInvPe
from relbench.modeling.gin import GIN
import numpy as np


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])


class MODEL_PE_LINK(Model):
    '''
        This model extends the RelBench base Model class by incorporating positional encodings
        into a heterogeneous GraphSAGE-based link prediction setup. The class supports positional encodings
        via SignNet-based encodings or PEARL-based encodings, handles large graphs via chunking,
        and integrates these into the GNN's forward pass. 
    '''
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        cfg=None,
        device=None
    ):
        super().__init__(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            out_channels=out_channels,
            aggr=aggr,
            norm=norm,
            shallow_list=shallow_list,
            id_awareness=id_awareness,
        )

        self.gnn = HeteroGraphSAGE_LINK(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
            cfg=cfg
        )

        self.device = device
        self.cfg = cfg
        self.pe_embedding = None

        if self.cfg.pe_type = 'signnet':
            gin = GIN(self.cfg.n_phi_layers, 1, self.cfg.hidden_phi_layers, self.cfg.pe_dims, self.create_mlp, bn=True)  
            rho = MLP_PE(self.cfg.pe_dims, 8 * self.cfg.pe_dims, self.cfg.hidden_phi_layers, 8, use_bn=True, activation='relu', dropout_prob=0.0)
            self.positional_encoding = SignInvPe(phi=gin, rho=rho)
            self.pe_embedding = torch.nn.Linear(8, self.cfg.node_emb_dims)
        else:
            phi = GINPhi(self.cfg.n_phi_layers, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn)          
            self.positional_encoding = K_PEARL_PE(phi, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, 
                            mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)
            self.num_samples = cfg.num_samples

    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None) -> MLP:
        print(in_dims)
        return MLP_PE(
            self.cfg.n_mlp_layers, in_dims, self.cfg.mlp_hidden_dims, out_dims, self.cfg.mlp_use_bn,
            self.cfg.mlp_activation, self.cfg.mlp_dropout_prob
         )

    '''
        This function computes positional encodings for PEARL. Batch is assumed to have the laplacian
        and edge index of the homogenous version of the batch subgraph.
    '''
    def compute_positional_encoding(self, batch: HeteroData) -> Tensor:
        # Due to the size of the graphs and/or samples, we may not have enough gpu space for an entire NxM or NxN input
        # We can therefore split the input up into [N, M//cfg.splits] or [N, N//cfg.splits] inputs and keep track of a
        # running sum in our positional encoder, passing each of these inputs cfg.splits times before returning
        # and resetting the running sum.
        if self.cfg.BASIS:
            N = batch.Lap[0].shape[0]
            basis_chunk = N // self.cfg.splits
            for k, start in enumerate(range(0, N, basis_chunk)):
                end = min(chunk_start + basis_chunk, N)
                W = torch.eye(N, device=self.device)[start:end]
                # If we are not at the last chunk of basis vectors, simply call the forward method to keep track of the 
                # running sum, otherwise get the positional encoding and reset the running sum by setting final=True
                if not (k == self.cfg.splits - 1): 
                    with torch.no_grad():
                        positional_encoding.forward(batch.Lap, [W], batch.edge_index, final=False)
                else:
                    PE = positional_encoding(batch.Lap, [W], batch.edge_index, final=True)
        else: 
            # If we are using random samples, we don't need to keep track of consecutive basis vectors and
            # we simply pass in self.num_samples // self.cfg.splits number of random vectors at a time.
            for k in range(self.cfg.splits):
                for i in range(len(batch.Lap)):
                    W = torch.randn(batch.Lap[i].shape[0], self.num_samples // self.cfg.splits).to(self.device)
                    W_list = [W] # Usually we append multiple but subgraph batch size is 1
                if k < self.cfg.splits - 1:
                    with torch.no_grad():
                        self.positional_encoding.forward(batch.Lap, [W], batch.edge_index, final=False)
                else:
                    PE = self.positional_encoding(batch.Lap, [W], batch.edge_index, final=True)
                torch.cuda.empty_cache()

        return PE

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        if self.cfg.pe_type == 'signnet':
            PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        else:
            PE = self.compute_positional_encoding(batch)
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict, 
            batch.edge_index_dict,
            PE, 
            batch.hom_to_het,
            batch.edge_index,
            batch
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    '''
        This forward function is the same as the one above but is implemented with ID-awareness. 
    '''
    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        if self.cfg.pe_type == 'signnet':
            PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        else:
            PE = self.compute_positional_encoding(batch)

        x_dict = self.encoder(batch.tf_dict)
        
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            PE, 
            batch.hom_to_het,
            batch.edge_index,
            batch
        )

        return self.head(x_dict[dst_table])
