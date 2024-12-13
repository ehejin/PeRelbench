import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from examples.model import MODEL_PE_LINK, Model_PEARL, Model_SIGNNET
from examples.text_embedder import GloveTextEmbedding
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import HeteroData, Data

from examples.config import merge_config

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
from torch_geometric.utils import to_scipy_sparse_matrix
import wandb


'''
    This function takes in a sparse laplacian matrix and returns the k-th eigenvectors and eigenvalues
    corresponding to the largest or smallest eigenvalues. If smallest is True, returns the smallest, 
    otherwise returns the largest.
'''
def get_eigen(laplacian, k, device, smallest):
    try: 
        if smallest:
            which = 'SM'
        else:
            which = 'LM'
        # Get the largest or smallest k eigenvalues and eigenvectors
        eigvals, eigvecs = eigsh(laplacian, k=k, which=which, maxiter=500000)  
        eigvals = torch.tensor(eigvals, device=device, dtype=torch.float32)
        eigvecs = torch.tensor(eigvecs, device=device, dtype=torch.float32)
    except:
        # If eigsh fails to converge, just compute using torch.linalg
        print("Eigsh failed to converge, using torch.linalg.eigh")
        laplacian = laplacian.toarray()
        dense_laplacian = torch.tensor(laplacian, device=device, dtype=torch.float32)
        eigvals, eigvecs = torch.linalg.eigh(dense_laplacian)
        if not smallest:
            eigvals = eigvals[-k:]  # Index the k largest eigenvalues and corresponding eigenvectors
            eigvecs = eigvecs[:, -k:]
        else:
            eigvals = eigvals[:k] 
            eigvecs = eigvecs[:, :k]
    
    return eigvals, eigvecs


class transform_LAP():
    '''
    This class transforms each heterogenous graph that is sampled into a 
    homogenous graph for calculating PE's, and stores the necessary graph info. 
    '''
    def __init__(self, pe_type, smallest, device, pe_dims):
        self.pe_type = pe_type
        self.pe_dims = pe_dims
        self.smallest = smallest
        self.device = device

    def __call__(self, hetero_data):
        # Map nodes (node type and local index) from the heterogenous graph 
        # to a global index in the homogenous graph
        het_to_hom = {} 
        type_index = 0 # 
        total_nodes = 0
        # Map back from the homogenous graph to the heterogenous node type and local index
        hom_to_het = {} 

        # For each node_type, we allocate a block of global indices in the homogenous graph
        for node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type]['n_id'].size(0)
            het_to_hom[node_type] = torch.arange(type_index, type_index + num_nodes)
            # For each node, we keep track of the reverse mapping
            for i in range(num_nodes):
                hom_to_het[type_index + i] = (node_type, i)
            type_index += num_nodes
            total_nodes += num_nodes
        
        edge_list = []

        # Convert edges in the heterogenous graph to an homogenous edge_index
        for edge_type in hetero_data.edge_types:
            source_type, _, dest_type = edge_type
            source_nodes, dest_nodes = hetero_data[edge_type].edge_index
            
            # Get the global homogenous node indices for heterogenous source and destination nodes
            hom_source = het_to_hom[source_type][source_nodes]
            hom_dest = het_to_hom[dest_type][dest_nodes]
            edge_list.append(torch.stack([hom_source, hom_dest], dim=0))
        
        # Concatenate edge indices across all edge types
        edge_index = torch.cat(edge_list, dim=1)

        # We compute and store the normalized Laplacian to compute our PE's
        edge_index, weights = get_laplacian(edge_index, normalization='sym')

        # If we are using SignNet PE's, we need the eigenvector and eigenvalues of the homogenous graph
        if self.pe_type == 'signnet':
            laplacian_sparse = to_scipy_sparse_matrix(edge_index, weights, num_nodes=total_nodes)
            eigenvalues, eigenvectors = get_eigen(laplacian_sparse, k=self.pe_dims, device=self.device, smallest=self.smallest) 
            hetero_data.Lambda = eigenvalues
            hetero_data.V = eigenvectors
        else: # We need the Laplacian for Pearl PE
            lap = to_dense_adj(edge_index, edge_attr=weights)
            hetero_data.Lap = lap

        # Store the graph information of our new homogenous graph.
        hetero_data.edge_index = edge_index
        hetero_data.num_nodes = total_nodes
        hetero_data.hom_to_het = hom_to_het

        return hetero_data


'''
    The rest of this script is adapted from RelBench examples/idgnn_link.py
'''
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument('--name', type=str, default=None)
parser.add_argument("--cfg", type=str, default=None, help="Path to PE cfg file")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument(
    "--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples")
)
args = parser.parse_args()

# Set up device for training
device = torch.device(f'cuda:{args.gpu_id}') 
print(device)
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

# Load configuration for PE model
cfg = merge_config(args.cfg)

# Initalize wandb if needed
if args.wandb:
    run = wandb.init(config=cfg, project='Relbench-LINK', name=args.name)

# Load dataset and task
dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

# Create Schema graph from the tables using fkeys and pkeys
stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

# Make the Schema graph
data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

# This controls how many neighbors we samples in the time computation graph
num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

# Create NeighborLoader objects for tain, val, and test. This samples a subgraph for each batch
loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    transform = transform_LAP(cfg.pe_type, cfg.smallest, device, cfg.pe_dims)
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        transform=transform
    )

# Load our RelBench model with the PE framework integrated
model = Model_PE_LINK(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=1,
    aggr=args.aggr,
    norm="layer_norm",
    id_awareness=True,
    cfg=cfg,
    device=device
).to(device)

print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)


def train() -> float:
    '''
        This function trains the model over one epoch, returning average training loss.
        Since this is a link prediction task, we use BCE loss.
    '''
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)

    # Loop through training batches
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)
        out = model.forward_dst_readout(
            batch, task.src_entity_table, task.dst_entity_table
        ).flatten()

        batch_size = batch[task.src_entity_table].batch_size

        # Get ground-truth labels
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target labels
        target = torch.isin(
            batch[task.dst_entity_table].batch
            + batch_size * batch[task.dst_entity_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Compute loss and update our parameters
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        optimizer.step()
        
        # Accumulate loss
        loss_accum += float(loss) * out.numel()
        count_accum += out.numel()

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    '''
        This function evaluates the model given a NeighborLoader with
        data to evlauate on. Returns the predicted top-k destination nodes for 
        each source node.
    '''
    model.eval()

    pred_list: list[Tensor] = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = (
            model.forward_dst_readout(
                batch, task.src_entity_table, task.dst_entity_table
            )
            .detach()
            .flatten()
        )
        batch_size = batch[task.src_entity_table].batch_size
        scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
        scores[
            batch[task.dst_entity_table].batch, batch[task.dst_entity_table].n_id
        ] = torch.sigmoid(out)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred

# Train and evaluate our model 
state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()

    # Evaluate the model at certain intervals
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        test_pred = test(loader_dict["test"])
        test_metrics = task.evaluate(test_pred)
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        # Log metrics
        val_metrics['Train loss'] = train_loss
        val_metrics['test_MAP'] = test_metrics['link_prediction_map']
        wandb.log(val_metrics)
        
        # Save the best model
        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict) # Load the best performing model for testing

# Final evaluation on validation and test splits
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")