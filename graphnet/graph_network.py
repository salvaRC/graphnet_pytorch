from typing import Dict, Sequence, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

from graphnet.constants import AggregationTypes, NODES, EDGES, GLOBALS
from graphnet.graph_network_block import GraphNetBlock


class GraphNetwork(pl.LightningModule):
    def __init__(self,
                 input_dim: Union[int, Dict[str, int]],
                 hidden_dims: Sequence[int],
                 senders: Sequence[int],
                 receivers: Sequence[int],
                 readout_which_output: Optional[str] = NODES,
                 update_mlp_n_layers: int = 1,
                 aggregator_funcs: Union[str, Dict[AggregationTypes, int]] = 'sum',
                 net_normalization: str = 'layer_norm',
                 residual: Union[bool, Dict[str, bool]] = True,
                 activation_function: str = 'Gelu',
                 output_activation_function: Optional[str] = None,
                 output_net_normalization: bool = True,
                 dropout: float = 0.0
                 ):
        """
        Args:
            input_dim: If a dict, it should have key-value pairs {GLOBALS: d_glob, EDGES: d_edges, NODES: d_nodes},
                            where d_* refers to the feature dimension of the respective graph component.
                       If an int, it is assumed that all graph components have the same feature input dimension.
            hidden_dims (list[int]): Defines the hidden dimensions (and depth) of the graph network
                                        (= the hidden dimensions of the 'processor' blocks).
            senders (list[int]): Defines the sender nodes, where senders_i is the index of the sending node of edge i
                                 Note 1: To get the senders, receivers defined by an adjacency matrix,
                                         call graphnet.utils.adj_to_sender_receivers(adjacency_matrix)
                                 Note 2: The graph/edge/sender-receiver structure can be changed at a later time by
                                       calling update_graph_structure(.) with a new pair of senders, receivers.
            receivers (list[int]): Defines the receiver nodes, where receivers_i is the index of the receiver node of edge i
                                   Note 1: To get the senders, receivers defined by an adjacency matrix,
                                           call graphnet.utils.adj_to_sender_receivers(adjacency_matrix)
                                   Note 2: The graph/edge/sender-receiver structure can be changed at a later time by
                                         calling update_graph_structure(.) with a new pair of senders, receivers.
            aggregator_funcs: If a dict, it should have key-value pairs {GLOBALS: agg_g, EDGES: agg_e, NODES: agg_n},
                                where agg_* refers to the aggregation function to be used for that graph component.
                              If a str, the same aggregator function is used for all graph components.
                              Must be one of {'sum', 'mean', 'max'}
            readout_which_output: Which graph component to return (default: nodes),
                                        must be on of {EDGES, NODES, GLOBALS, 'graph', None}
                                  If None or 'graph', the whole graph is returned.
        """
        super().__init__()
        self.save_hyperparameters(ignore="verbose_mlp")
        assert len(self.hparams.hidden_dims) >= 1
        assert update_mlp_n_layers >= 1

        graphnet_layers = []
        in_dim = input_dim
        dims = [input_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            out_activation_function = output_activation_function if i == len(dims) - 1 else activation_function
            out_net_norm = output_net_normalization if i == len(dims) - 1 else True
            graphnet_layers += [
                GraphNetBlock(
                    in_dims=in_dim,
                    out_dims=dims[i],
                    senders=senders,
                    receivers=receivers,
                    n_layers=update_mlp_n_layers,
                    residual=residual,
                    net_norm=net_normalization,
                    activation=activation_function,
                    dropout=dropout,
                    output_normalization=out_net_norm,
                    output_activation_function=out_activation_function,
                    aggregator_funcs=aggregator_funcs,
                )]
            in_dim = dims[i]

        self.layers: nn.ModuleList[GraphNetBlock] = nn.ModuleList(graphnet_layers)
        self.output_type = readout_which_output
        if self.output_type not in [NODES, EDGES, GLOBALS, 'graph', None]:
            raise ValueError("Unsupported argument for GraphNetwork `output_type`", readout_which_output)

    @property
    def n_edges(self):
        return self.layers[0].n_edges

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_graph_structure(self, senders: Sequence[int], receivers: Sequence[int]) -> None:
        for layer in self.layers:
            layer.update_graph_structure(senders, receivers)

    def forward(self, X: Dict[str, Tensor]):
        """
        X:
            Dict with key-values {GLOBALS: x_glob, EDGES: x_edges, NODES: x_nodes},
             where x_*** are the corresponding features.
        """
        graph_new = self.update_graph(X)
        if self.output_type is not None and self.output_type != 'graph':
            graph_component = graph_new[self.output_type]
            return graph_component  # .reshape(graph_component.shape[0], -1)
        else:
            return graph_new

    def update_graph(self, X: Dict[str, Tensor]) -> Dict[str, Tensor]:
        graph_net_input = X
        graph_new = self.layers[0](graph_net_input)
        for graph_net_block in self.layers[1:]:
            graph_new = graph_net_block(graph_new)

        return graph_new

