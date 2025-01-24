# RelBench w/ Learnable Positional Encodings

<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

**This repository is a fork of the [official RelBench repository](https://github.com/snap-stanford/relbench).** RelBench is a benchmark for deep learning on relational databases. The original repository and documentation provide instructions, data, and a framework for training and evaluating models on multiple real-world relational tasks. This fork incorporates my modifications detailed below.

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

---

## Attribution

This repo also contains modules and models adapted from the [**SPE** repo](https://github.com/Graph-COM/SPE) by Huang et al. 

## Changes in This Fork

**This fork contains SignNet and PEARL positional encoding models that are incorporated into the RelBench model.**  
   - modeling/pe.py contains the positional encoder models.
   - examples/model.py contains the edited version of model to support PE's, namely the MODEL_PE_LINK class.
   - modeling/mlp.py and gin.py contain helper modules.
   - modeling/nn.py contains the edited HeteroGraphSAGE model for PE integration.
   - examples/configs contain configs that are use for the PE models, and merged with examples/config.py
   - the training script with dataset preprocessing is in examples/PE_idgnn_link.py


**To run the experiments with PEs, run the following commands:**
```bash
# To run SignNet-8L
python -m examples.PE_idgnn_link --dataset rel-stack --task user-post-comment --batch_size 20 --epochs 20 --cfg ./examples/configs/signnet-8L.yaml
# To run SignNet-8S
python -m examples.PE_idgnn_link --dataset rel-stack --task user-post-comment --batch_size 20 --epochs 20 --cfg ./examples/configs/signnet-8S.yaml
# To run R-PEARL
python -m examples.PE_idgnn_link --dataset rel-stack --task user-post-comment --batch_size 20 --epochs 20 --cfg ./examples/configs/R-PEARL.yaml
# To run B-PEARL
python -m examples.PE_idgnn_link --dataset rel-stack --task user-post-comment --batch_size 20 --epochs 20 --cfg ./examples/configs/B-PEARL.yaml
```
**To run on the post-post-related task, simply use --task post-post-related instead.**

---

## Overview

**What is RelBench?**  
RelBench is a benchmark for end-to-end representation learning on relational databases. It offers:

To learn more, see the [**Benchmark Paper**](https://arxiv.org/abs/2407.20060) or visit the [**website**](https://relbench.stanford.edu).

---

## Installation

Install the core functionality of RelBench via `pip`:

```bash
pip install relbench

