import os
import argparse
from yacs.config import CfgNode as CN

cfg = CN()

# Architecture of all MLPs in our model
cfg.n_mlp_layers = 2 
cfg.mlp_hidden_dims = 40
cfg.mlp_use_bn = True
cfg.mlp_activation = 'relu'
cfg.mlp_dropout_prob = 0

# Architecture of GINPhi
cfg.n_phi_layers = 7
cfg.hidden_phi_layers = 40
cfg.pe_dims = 40  # Dimension of our PE's
cfg.BASIS = False # Whether we are using B-PEARL or R-PEARL (not applicable to SignNet)
cfg.RAND_k = 10 # Maximum order of the laplacian filter

# Architecture of our MLP applied after GINPhi (not applicable to SignNet)
cfg.RAND_mlp_nlayers = 1
cfg.RAND_mlp_hid = 40
cfg.RAND_act = 'relu'
cfg.RAND_mlp_out = 40

# Embedding dimensions of our original node features (output of ResNet Tabular)
cfg.node_emb_dims = 128 
cfg.num_samples = 80 # Number of samples for R-PEARL
cfg.PE1 = False
cfg.splits = 1 
cfg.smallest = False # Whether we take the smallest or largest eigenvectors for SignNet
cfg.pe_type = 'signnet'

def load_config_from_file(config_path):
    if config_path and os.path.exists(config_path):
        cfg.merge_from_file(config_path)

def merge_config(path):
    if path is not None:
        load_config_from_file(path)
    return cfg