# Graph Network in Pytorch(-Lightning)
<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.9.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
![CC BY 4.0][cc-by-image]

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This is a Pytorch implementation of graph networks, a graph-based neural network proposed in 
the paper [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261) by Battaglia et al.
The official code implementation (based on Tf and Sonnet) can be found [here](https://github.com/deepmind/graph_nets).
It expects the input to be a dictionary mapping to the node, edge and global input features.
See the [run_demo.py](run_demo.py) script for a synthetic example.

## Getting Started

    conda env create -f env_gn.yml   # create new environment will all dependencies
    conda activate graphnet  # activate the environment called 'graphnet'
    python run_demo.py
    