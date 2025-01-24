from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroGraphSAGE_LINK, HeteroTemporalEncoder, HeteroEncoder_PEARL#, HeteroGraphSAGE_PEARL
from relbench.modeling.mlp import MLP as MLP_PE
from relbench.modeling.pe import K_PEARL_PE, GINPhi, SignInvPe
from relbench.modeling.gin import GIN
import numpy as np


class Model(torch.nn.Module):
    '''
        This model is from the original RelBench repo. It supports tasks like link prediction
        which requires ID-GNN base model. It contains a HeteroEncoder that encodes node features
        using column information, HeteroTemporalEncoder, which captures temporal dependencies,
        HeteroGraphSAGE, which performs message passing, and an MLP head for task-specific predictions.
        If id_awareness is true, will support the ID-GNN.
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
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        # Processes node features using tabular data
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        # Captures time-dependent relationships
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        # Heterogenous Message Passing
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        # Task-specific MLP
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        # Shallow embeddings for node types
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        # Support for ID-awareness
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

    '''
        This method takes in a batch of hetero graph data and an entity table that indicates
        for which predictions to make. Returns a tensor of predictions.
    '''
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict) # Get node features

        # Get temporal encoding
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        # Add temporal embeddings and encoded node features
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        # Add the shallow embeddings to node features if they exist
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    '''
        This method is the same as the forward method above, except it adds ID-awareness.
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
        x_dict = self.encoder(batch.tf_dict)

        # Add ID-awareness to the root node!
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


class Model_SIGNNET(nn.Module):
    """
    A SignNet-based GNN model with ID-awareness
    """

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict["StatType", Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List["NodeType"] = [],
        id_awareness: bool = False,
        cfg=None,
        device=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.encoder = HeteroEncoder_PEARL(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers
        )
        self.head = MLP(
            in_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        gin = GIN(
            num_layers=self.cfg.n_phi_layers,
            in_channels=1,
            hidden_channels=self.cfg.hidden_phi_layers,
            out_channels=4,
            create_mlp=self.create_mlp,
            bn=True
        )
        rho = MLP_PE(
            in_channels=4,
            hidden_channels=4 * self.cfg.pe_dims,
            num_layers=self.cfg.hidden_phi_layers,
            out_channels=self.cfg.pe_dims,
            use_bn=True,
            activation="relu",
            dropout_prob=0.0
        )
        self.positional_encoding = SignInvPe(phi=gin, rho=rho)
        self.pe_embedding = nn.Linear(self.cfg.pe_dims, self.cfg.node_emb_dims)
        self.embedding_dict = ModuleDict(
            {
                node_type: Embedding(data.num_nodes_dict[node_type], channels)
                for node_type in shallow_list
            }
        )
        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = nn.Embedding(1, channels)
        self.reset_parameters()

    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None) -> nn.Module:
        return MLP_PE(
            num_layers=self.cfg.n_mlp_layers,
            in_channels=in_dims,
            hidden_channels=self.cfg.mlp_hidden_dims,
            out_channels=out_dims,
            use_bn=self.cfg.mlp_use_bn,
            activation=self.cfg.mlp_activation,
            dropout_prob=self.cfg.mlp_dropout_prob
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
        nn.init.xavier_uniform_(self.pe_embedding.weight)

    def forward(
        self,
        batch: HeteroData,
        entity_table: "NodeType",
        W=None,
        print_emb: bool = False,
        device=None
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time

        PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        reverse_node_mapping = batch.reverse_node_mapping
        projected_PE = self.pe_embedding(PE)
        PE_dict = {}

        last_node_idx = -1

        for homogeneous_idx, pos_encoding in enumerate(projected_PE):
            node_type, node_idx = reverse_node_mapping[homogeneous_idx]
            if node_type not in PE_dict:
                last_node_idx = -1
                PE_dict[node_type] = pos_encoding.unsqueeze(dim=0)
            else:
                PE_dict[node_type] = torch.cat(
                    (PE_dict[node_type], pos_encoding.unsqueeze(dim=0)), dim=0
                )
            last_node_idx = node_idx
        x_dict = self.encoder(batch.tf_dict, PE_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] += rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] += embedding(batch[node_type].n_id)
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict
        )
        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: "NodeType",
        dst_table: "NodeType"
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be True to use forward_dst_readout."
            )
        seed_time = batch[entity_table].seed_time
        PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        reverse_node_mapping = batch.reverse_node_mapping
        projected_PE = self.pe_embedding(PE)
        PE_dict = {}
        last_node_idx = -1

        for homogeneous_idx, pos_encoding in enumerate(projected_PE):
            node_type, node_idx = reverse_node_mapping[homogeneous_idx]
            if node_type not in PE_dict:
                last_node_idx = -1
                PE_dict[node_type] = pos_encoding.unsqueeze(dim=0)
            else:
                PE_dict[node_type] = torch.cat(
                    (PE_dict[node_type], pos_encoding.unsqueeze(dim=0)), dim=0
                )
            last_node_idx = node_idx
        x_dict = self.encoder(batch.tf_dict, PE_dict)

        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] += rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] += embedding(batch[node_type].n_id)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
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

        # Our edited hetero GNN that supports PE's in the forward function
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

        # Initialize our PE model depending on if we use the PEARL or signnet framework
        if self.cfg.pe_type == 'signnet':
            gin = GIN(self.cfg.n_phi_layers, 1, self.cfg.hidden_phi_layers, 4, self.create_mlp, bn=True)  
            rho = MLP_PE(4, 4 * self.cfg.pe_dims, self.cfg.hidden_phi_layers, self.cfg.pe_dims, use_bn=True, activation='relu', dropout_prob=0.0)
            self.positional_encoding = SignInvPe(phi=gin, rho=rho)
            self.pe_embedding = torch.nn.Linear(8, self.cfg.node_emb_dims)
        else:
            phi = GINPhi(self.cfg.n_phi_layers, self.cfg.RAND_mlp_out, self.cfg.hidden_phi_layers, self.cfg.pe_dims, 
                                self.create_mlp, self.cfg.mlp_use_bn, basis=self.cfg.BASIS)          
            self.positional_encoding = K_PEARL_PE(phi, k=cfg.RAND_k, mlp_nlayers=cfg.RAND_mlp_nlayers, 
                            mlp_hid=cfg.RAND_mlp_hid, spe_act=cfg.RAND_act, mlp_out=cfg.RAND_mlp_out)
            self.num_samples = cfg.num_samples

    def create_mlp(self, in_dims: int, out_dims: int, use_bias=None) -> MLP:
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

        # Add positional encodings. If we use SignNet, simply pass in the eigenvectors to the model
        if self.cfg.pe_type == 'signnet':
            PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        else: # For pearl, we need to generate the random and basis nodes first
            PE = self.compute_positional_encoding(batch)
        x_dict = self.encoder(batch.tf_dict)

        # Get temporal info
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        # Add temporal information to the node features
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        
        # Also add shallow embeddings
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        # pass thru our edited gnn with PE support
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
        # Add positional encodings. If we use SignNet, simply pass in the eigenvectors to the model
        if self.cfg.pe_type == 'signnet':
            PE = self.positional_encoding(None, batch.V, batch.edge_index, batch=None)
        else: # For pearl, we need to generate the random and basis nodes first
            PE = self.compute_positional_encoding(batch)

        x_dict = self.encoder(batch.tf_dict)
        
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        # Add temporal information to the node features
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        
        # Also add shallow embeddings
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


