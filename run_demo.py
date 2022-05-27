import torch

from graphnet.constants import NODES, EDGES, GLOBALS
from graphnet.graph_network import GraphNetwork

if __name__ == '__main__':
    num_edges = 50
    num_nodes = 49
    num_features_edges = 5
    num_features_nodes = 64
    num_features_globals = 16  # globals is same as vector u
    batch_size = 128

    # generate a random graph structure
    # To get the senders, receivers defined by an adjacency matrix,
    # call senders, receivers = graphnet.utils.adj_to_sender_receivers(adjacency_matrix)
    senders = torch.randint(0, num_nodes, num_edges)
    receivers = torch.randint(0, num_nodes, num_edges)

    # initialize the graph network
    graph_net = GraphNetwork(
        input_dim={NODES: num_features_nodes, EDGES: num_features_edges, GLOBALS: num_features_globals},
        hidden_dims=[128, 128],  # 2 layer graphnet (= 2 processor blocks)
        senders=senders,
        receivers=receivers,
        update_mlp_n_layers=1,
        readout_which_output=NODES,
        residual=True,
        activation_function='gelu',
        output_net_normalization=True
    )

    # generate random input data
    E = torch.randn((batch_size, num_edges, num_features_edges))
    V = torch.randn((batch_size, num_nodes, num_features_nodes))
    U = torch.randn((batch_size, num_features_globals))
    X_in = {NODES: V, EDGES: E, GLOBALS: U}

    # forward it through the model
    X_out = graph_net(X_in)

    print(X_out.shape)  # use 'readout_which_output' arg to control which part of the graph to return
