"""
Author: Salva Rühling Cachay
"""
import logging
import os

from typing import Union, Sequence, List, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only


def get_activation_function(name: str, functional: bool = False, num: int = 1):
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "identity": nn.Identity(),
                None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu, 'prelu': nn.PReLU(),
                }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU(),
                }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_normalization_layer(name, dims, num_groups=None, *args, **kwargs):
    if not isinstance(name, str) or name.lower() == 'none':
        return None
    elif 'batch' in name:
        return nn.BatchNorm1d(num_features=dims, *args, **kwargs)
    elif 'layer' in name:
        return nn.LayerNorm(dims, *args, **kwargs)
    elif 'inst' in name:
        return nn.InstanceNorm1d(num_features=dims, *args, **kwargs)
    elif 'group' in name:
        if num_groups is None:
            num_groups = int(dims / 10)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)


def adj_to_sender_receivers(
        adj: Union[torch.Tensor, np.ndarray]
) -> (Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]):
    """
    Args:
        adj: a (N, N) adjacency matrix, where N is the number of nodes
    Returns:
        A tuple of (senders, receivers) node indices
    """
    edge_tuples = torch.nonzero(adj, as_tuple=True) if torch.is_tensor(adj) else np.nonzero(adj)
    edge_src = edge_tuples[0].unsqueeze(0) if torch.is_tensor(adj) else np.expand_dims(edge_tuples[0], axis=0)
    edge_dest = edge_tuples[1].unsqueeze(0) if torch.is_tensor(adj) else np.expand_dims(edge_tuples[1], axis=0)
    return edge_src, edge_dest


def adj_to_edge_indices(adj: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        adj: a (N, N) adjacency matrix, where N is the number of nodes
    Returns:
        A (2, E) array, edge_idxs, where E is the number of edges,
                and edge_idxs[0],  edge_idxs[1] are the source & destination nodes, respectively.
    """
    edge_src, edge_dest = adj_to_sender_receivers(adj)
    if torch.is_tensor(adj):
        edge_idxs = torch.cat((edge_src, edge_dest), dim=0)
    else:
        edge_idxs = np.concatenate((edge_src, edge_dest), axis=0)
    return edge_idxs


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def set_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_seed(seed, device='cuda'):
    import random
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
