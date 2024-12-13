# RelBench (Forked and Enhanced)

<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

**This repository is a fork of the [official RelBench repository](https://github.com/snap-stanford/relbench).** RelBench is a benchmark for deep learning on relational databases. The original repository and documentation provide instructions, data, and a framework for training and evaluating models on multiple real-world relational tasks. This fork incorporates my modifications and enhancements detailed below.

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

---

## Changes in This Fork

This fork contains the following modifications:

1. **Added SignNet and PEARL positional encoding models for use:**  
   - To run experiments run this:
   - modeling/pe.py contains the positional encoder models
   - examples/model.py contains the edited version of model to support PE's, namely the MODEL_PE_LINK class
   - modeling/mlp.py and gin.py contain helper models
   - modeling/nn.py containst the edited HeteroGraphSAGE model for PE integration

This repo also contains models adapted from [**SPE**](https://github.com/Graph-COM/SPE)
---

## Overview

**What is RelBench?**  
RelBench is a benchmark for end-to-end representation learning on relational databases. It offers:

- **Realistic, large-scale relational datasets** spanning domains including medical, social networks, e-commerce, and sports.
- **Multiple prediction tasks** (30 in total) defined for each dataset.
- **Framework-agnostic tools** for data loading, standardized evaluation, and reproducible experimentation.
- **Graph-based modeling support**: transform relational databases into graphs suitable for Graph Neural Network (GNN) models.

To learn more, see the [**Benchmark Paper**](https://arxiv.org/abs/2407.20060) or visit the [**website**](https://relbench.stanford.edu).

---

## Installation

Install the core functionality of RelBench via `pip`:

```bash
pip install relbench

