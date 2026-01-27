"""

The pre-processing required to run EB-GNN.
Basically, we pre-compute edge adjacencies according to alpha, beta and gamma aggergations 

"""

from collections import defaultdict

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, is_undirected

class FastEdgeGraph(Data):
    def __inc__(self, key, value, store):
        if key == "edges_to_target":
            return torch.tensor(self.edge_index.size(1))
        elif key == "edge_batch":
            return 1  # edge_batch is a batch index
        elif key in ["wl_mapping",  "wr_mapping"]:
            # w1_mapping has shape [num_mappings, 2]
            return torch.tensor([self.edge_index.size(1), self.edge_index.size(1)])
        elif key == "wt_mapping":
            # wt_mapping has shape [num_mappings, 3]
            return torch.tensor([self.edge_index.size(1), self.edge_index.size(1), self.edge_index.size(1)])
        elif key == "nc_mapping":
            # nc_mapping has shape [num_mappings, 4] with: node_idx, node_idx, node_idx, edge_idx
            return torch.tensor([self.num_nodes, self.num_nodes, self.num_nodes, self.edge_index.size(1)])
        else:
            return super().__inc__(key, value, store)
        
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "edges_to_target":
            return 0
        elif key in ["wl_mapping", "wr_mapping", 'wt_mapping', "nc_mapping"]:
            return 0  # Concatenate along dimension 0
        return super().__cat_dim__(key, value, *args, **kwargs)
    
from collections import defaultdict

class EBGNNTransform(BaseTransform):
    r""" 
    Select only one target from data.y
    """
    def __init__(self):
        pass

    def __call__(self, data: Data):
        edge_index = data.edge_index
        
        wl_tensor, wr_tensor, wt_tensor, edge_batch = EBGNN_transform(edge_index)
            
        kwargs = dict(
            y=data.y,
            x=data.x,
            edge_index=edge_index,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
            wl_mapping=wl_tensor,
            wr_mapping=wr_tensor,
            wt_mapping=wt_tensor,
            edge_batch=edge_batch,
        )    
        
        if hasattr(data, "edges_to_target"):
            kwargs["edges_to_target"] = data.edges_to_target

        out = FastEdgeGraph(**kwargs)
        print(out)
        return out

def compute_triangles(edge_index, neighbors, do_test = False):
    """
    Chiba–Nishizeki algorithm for triangle counting
    runs in O(E * arboricity)
    """

    # For undirected graphs, these can have different sizes
    deg1 = degree(edge_index[0])
    deg2 = degree(edge_index[1])

    n = max(deg1.numel(), deg2.numel())

    # Combine into a single deg tensor
    # Technically this is not the correct degree but it is correct for determining whic node has higher degree
    deg = F.pad(deg1, (0, n - deg1.numel())) + F.pad(deg2, (0, n - deg2.numel()))
    
    # Create a set of edges for O(1) lookup
    # Python  sets are implemented as hash table with on average O(1) lookup: 
    #   https://stackoverflow.com/questions/7351459/time-complexity-of-python-set-operations
    edge_set = set()
    for u, v in edge_index.t().tolist():
        
        # Process edges as (u,v) where u has the lower degree + arbitrary tie breaking
        if deg[u] < deg[v] or (deg[u] == deg[v] and u < v):
            edge_set.add((u,v))
        else:
            edge_set.add((v,u))
    
    def is_edge(u, v):
        """O(1) check if edge exists in the graph."""
        return (u,v) in edge_set or (v,u) in edge_set
    
    # triangles[(u,v)] = [x, ...] where (u,v,x) is a triangle
    triangles = defaultdict(lambda: [])
 
    for u, v in edge_set:
        for neighbor in neighbors[u]:
            if is_edge(v,neighbor):
                # Triangle found
                triangles[(u, v)] += [neighbor]
                triangles[(v, u)] += [neighbor]

    # print("#Triangles:", sum(list(map(lambda ls: len(ls), triangles.values())))/6)
    # quit() 
    if do_test:
        print("Checking if results are correct")
        print("#Triangles:", sum(list(map(lambda ls: len(ls), triangles.values())))/6)
        for src, dst in edge_index.t().tolist():
            assert set(neighbors[src].intersection(neighbors[dst])) == set(triangles[(src,dst)])
        print("Passed")
        
    return triangles
    
def edge_idx_lookup_with_rerouting_for_directed_graphs(idx, edge_idx_dict):
    """
    For undirect graphs, we can have the situation where a certain edge direction is not part of the graph.
    If this edge is required for a certain operation, we instead use the other direction
    """

    u,v = idx
    if (u,v) in edge_idx_dict:
        return edge_idx_dict[(u, v)]
    else:
        return edge_idx_dict[(v, u)]

def EBGNN_transform(edge_index, do_test = False):
    """
    This is designed to work purely on torch tensors, so it can be used in frameworks that do not use PyG (e.g. torchdrug)
    """

    is_directed_graph = not is_undirected(edge_index)
    
    neighbors = defaultdict(set)
    for src, dst in edge_index.t().tolist():
        neighbors[int(src)].add(int(dst))
 
    triangles = compute_triangles(edge_index, neighbors, do_test=do_test)
    
    # edge_idx_dict: maps (u, v) -> index of this edge
    edge_idx_dict = {}
    
    for idx in range(edge_index.shape[1]):
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        edge_idx_dict[(u, v)] = idx

    get_edge_idx = lambda idx: edge_idx_lookup_with_rerouting_for_directed_graphs(idx, edge_idx_dict) 

    # [Wl] 2-Tuple: idx(u, w), idx(u, v)
    wl_mapping = []
    
    # [Wt] 3-Tuple: idx(u,w), idx(v, w), idx(u, v)
    wt_mapping = []
    
    # [Wr] 2-Tuple: idx(v,w), idx(u,v)
    wr_mapping = []

    # The idx of the node from which to use the feature
    for idx in range(edge_index.shape[1]):
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])

        Nu, Nv = neighbors[u], neighbors[v]
            
        # For beta aggregations
        for w in triangles[(u,v)]:
            wt_mapping.append([get_edge_idx((u, w)),
                                get_edge_idx((v, w)),
                                idx])
        
        if do_test:
            assert set(Nu.intersection(Nv)) == set(triangles[(u,v)])
        
        # For alpha aggregation:
        for w in (Nu):
            wl_mapping.append([get_edge_idx((u,w)),
                                idx])
            
        # For gamma aggregation:
        for w in (Nv):
            wr_mapping.append([get_edge_idx((v, w)),
                                idx])

                
    if len(wt_mapping) == 0:
        wt_tensor = torch.zeros((0,3), dtype=torch.long)
    else:
        wt_tensor =  torch.tensor(wt_mapping, dtype=torch.long)
        
    if len(wl_mapping) == 0:
        wl_tensor = torch.zeros((0,2), dtype=torch.long)
    else:
        wl_tensor =  torch.tensor(wl_mapping, dtype=torch.long)
        
    if len(wr_mapping) == 0:
        wr_tensor = torch.zeros((0,2), dtype=torch.long)
    else:
        wr_tensor =  torch.tensor(wr_mapping, dtype=torch.long)
        
    device = edge_index.device
    return wl_tensor.to(device), wr_tensor.to(device), wt_tensor.to(device), torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)