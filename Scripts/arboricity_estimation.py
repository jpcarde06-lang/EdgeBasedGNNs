import os
import numpy as np

from torch_geometric.datasets import QM9, MalNetTiny
from torch_geometric.utils import to_networkx
import networkx as nx

from Misc.QM_dataset import QM_Dataset
from Misc.config import config

qm9_dataset = QM9(root=os.path.join(config.DATA_PATH, "QM9", "Original"))
malnet_dataset = MalNetTiny(root=os.path.join(config.DATA_PATH, "MalNetTiny"), split=None)
qmd_dataset = QM_Dataset(root=os.path.join(config.DATA_PATH, "QMD"), target="mulliken_dipole_tot")

for dataset in [qm9_dataset, malnet_dataset, qmd_dataset]:
    print(dataset)
    
    degeneracy_ls = []
    
    for idx, graph in enumerate(dataset):
        print(f"\r{idx+1}/{len(dataset)}", end = "")
        
        G_nx = to_networkx(graph, to_undirected=True)
        G_nx.remove_edges_from(nx.selfloop_edges(G_nx))
        
        core_numbers = nx.core_number(G_nx)  
        degeneracy = max(core_numbers.values())
        degeneracy_ls.append(degeneracy)
        
    print("\n")
    print("max:", max(degeneracy_ls))
    print("mean:", np.mean(degeneracy_ls))
    print("std:", np.std(degeneracy_ls))