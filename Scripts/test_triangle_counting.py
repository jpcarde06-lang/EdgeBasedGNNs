"""
Test whether our implemtation of Chiba–Nishizeki gives the same results as the more intuitive implementation
"""

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from Misc.EBGNN_transform import EBGNN_transform


nr_graphs = 10

for seed in range(nr_graphs):  
    print(f"{seed+1}/{nr_graphs}")
    G = nx.erdos_renyi_graph(n=2000, p=0.08,seed=seed)
    data = from_networkx(G)
    print(data)
    EBGNN_transform(data.edge_index, do_test = True)