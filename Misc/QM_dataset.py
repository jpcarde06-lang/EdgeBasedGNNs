import ast
import tarfile
import os

import pickle

import torch
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import PeriodicTable

from torch_geometric.utils import from_smiles
from torch_geometric.data import Data

import torch
from torch_geometric.data import InMemoryDataset, download_url

from Misc.utils import PredictionType

def QM_task_level(target):
    if target in ["bond_length_matrix", "bond_index_matrix", "mulliken_condensed_charge_matrix", "natural_ionicity"]:
        return PredictionType.UNDIR_EDGE_PREDICTION
    elif "homo" in target or target in ["IP", "EA", "mulliken_dipole_tot", "mulliken_quadrupoles"]:
        return PredictionType.GRAPH_PREDICTION
    else:
        return PredictionType.NODE_PREDICTION

class UndirEdgeTask(Data):
    def __inc__(self, key, value, store):
        if key == "edges_to_target":
            return torch.tensor(self.edge_index.size(1))
        else:
            return super().__inc__(key, value)
        
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "edges_to_target":
            return 0
        else:
           return super().__cat_dim__(key, value, *args, **kwargs)

class QM_Dataset(InMemoryDataset):
    def __init__(self, root, level="wb97xd", target="bond_index_matrix", scaffold=False, transform=None, pre_transform=None, pre_filter=None):
        self.level = level
        self.target = target
        self.scaffold = False
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return 1

    @property
    def num_tasks(self):
        return 1

    @property
    def raw_file_names(self):
        return ['data.tar.gz', 'splits.tar.gz',]

    @property
    def processed_file_names(self):
        return [f'data_{self.level}_{self.target}.pt']

    def download(self):
        # Download to `self.raw_dir`.

        # Data
        tar_path = download_url("https://zenodo.org/records/10668491/files/data.tar.gz", self.raw_dir)
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=self.raw_dir)

        # Splits
        tar_path = download_url("https://zenodo.org/records/10668491/files/splits.tar.gz", self.raw_dir)
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=self.raw_dir)


    def process(self):
        # Read data into huge `Data` list.

        df = pd.read_csv(os.path.join(self.raw_dir, "data", f"{self.level}.csv"))
        data_list = []

        for idx in range(len(df)):
            smiles = df['smiles'].iloc[idx]
            data = from_smiles(smiles, with_hydrogen=True)     
            task_level = QM_task_level(self.target)
            
            # Edge-level tasks
            if task_level == PredictionType.UNDIR_EDGE_PREDICTION:
                matrix = np.array(ast.literal_eval(df[self.target].iloc[idx]))
                
                # Get a tensor that gives me the index for both edge directions
                
                edge_index = data.edge_index
                
                # (u, v) -> (idx of (u,v), idx of (v,u)) for u > v
                look_up_dict = {}
                
                for idx in range(edge_index.shape[1]):
                    u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
                    
                    # No self edges!
                    assert u != v
                    
                    if u > v:
                        continue
                    
                    look_up_dict[(u,v)] = [idx]
                    
                for idx in range(edge_index.shape[1]):
                    u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
                    
                    if u < v:
                        continue
                    
                    look_up_dict[(v,u)].append(idx)
                  
                
                edges_to_target = []
                y = []
                
                for edge, indices in look_up_dict.items():
                    u, v = edge[0], edge[1]
                    edges_to_target.append(torch.tensor(indices, dtype=torch.long))
                    y.append(torch.tensor(matrix[edge[0], edge[1]], dtype=torch.float32))
                    
                    # Check if the edge property is symmetric
                    assert matrix[edge[0], edge[1]] == matrix[edge[1], edge[0]]
                             
                edges_to_target = torch.stack(edges_to_target)       
                y = torch.stack(y)       

                # Wrap in a special Data object to ensure that edges_to_target works with batching
                data = UndirEdgeTask(x = data.x,  edge_index= data.edge_index, edge_attr = data.edge_attr, edges_to_target=edges_to_target,  y=y, smiles=data.smiles)
                assert len(matrix) == data.x.shape[0]
                
            # Graph-level tasks
            elif task_level == PredictionType.GRAPH_PREDICTION:
                y =  torch.tensor(df[self.target].iloc[idx], dtype=torch.float32)
                data.y = y
                
            else:
                raise Exception("Not implemented")
                
            data_list.append(data)
            

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self):
        if self.scaffold:
            filename = "scaffold_split_indices.pckl"
        else:
            filename = "random_split_indices.pckl"

        pth = os.path.join(self.raw_dir, "splits", filename)
        with open(pth, 'rb') as f:
            splits = pickle.load(f)

        return {
            "train": splits[0],
            "valid": splits[1],
            "test": splits[2]
        }
