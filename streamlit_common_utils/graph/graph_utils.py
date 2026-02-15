# Imports
import numpy as np
import networkx as nx

# Main Functions
# Basic Dataset Functions
def generate_networkx_graph_from_adjacency_matrix(Adj) -> nx.Graph:
    '''
    Generate Graph - Generates the NetworkX Graph Object from adjacency matrix

    Args:
        Adj (array): Input adjacency matrix

    Returns:
        G (Graph): Generated NetworkX Graph Object
    '''
    # Init Graph
    G = nx.Graph()

    # Add Nodes
    for i in range(Adj.shape[0]):
        G.add_node(i)

    # Add Edges
    for i in range(Adj.shape[0]):
        for j in range(Adj.shape[1]):
            if not (Adj[i, j] == np.inf):
                G.add_edge(i, j, weight=Adj[i, j])

    return G