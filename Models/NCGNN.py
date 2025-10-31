from collections import defaultdict

import torch
from torch_geometric.nn import MessagePassing, GINEConv, GATv2Conv
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.data import Data

from Models.utils import get_pooling_fct, get_activation, get_mlp
from Misc.utils import PredictionType
from Misc.EBGNN_trafo import FastEdgeGraph    

class NCGNNTransform(BaseTransform):
    r""" 
    Pre-compute triangles for NC-GNN triangle message passing
    """
    def __init__(self):
        pass

    def __call__(self, data: Data):

        neighbors = defaultdict(set)
        for src, dst in data.edge_index.t().tolist():
            neighbors[int(src)].add(int(dst))

        edge_to_idx = {}
        for idx, (src, dst) in enumerate(data.edge_index.t().tolist()):
            src, dst = int(src), int(dst)
            neighbors[src].add(dst)
            neighbors[dst].add(src)  # if undirected
            edge_to_idx[(src, dst)] = idx
            edge_to_idx[(dst, src)] = idx  # if undirected

        # Each entry [x,y,z] corresponds to a triangle between these two nodes.
        # Note that this will also contain [y, x, z] and [z, x, y] (assuming x < y < z)
        nc_mapping = []

        # This code is highly inefficient, but for only comparing predictive performance to NC-GNN it works
        for node in range(int(torch.max(data.edge_index)) + 1):
            for neighbor1 in neighbors[node]:
                for neighbor2 in neighbors[node]:

                    # Avoid "degenerate" triangles
                    if neighbor1 == node or neighbor2 == node or neighbor1 == neighbor2:
                        continue

                    # Needs to form a triangle
                    if neighbor1 not in neighbors[neighbor2]:
                        continue

                    # Do not overcount triangles
                    if neighbor2 > neighbor1:
                        continue

                    edge_idx = edge_to_idx.get((neighbor1, neighbor2))
                    nc_mapping.append((node, neighbor1, neighbor2, edge_idx))

        kwargs = dict(
            y=data.y,
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
            nc_mapping=torch.tensor(nc_mapping, dtype=torch.long),
        )    
        
        if hasattr(data, "edges_to_target"):
            kwargs["edges_to_target"] = data.edges_to_target
        
        return FastEdgeGraph(**kwargs)

class NCGNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer, emb_dim, 
                    gnn_type, residual, drop_ratio, JK, graph_pooling,
                    node_encoder, edge_encoder, num_mlp_layers, activation, prediction_type, parallel_prediction):
        """
        Message passing graph neural network.
        """
        super(NCGNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.residual = residual
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.activation = get_activation(activation)
        self.prediction_type = prediction_type
        self.parallel_prediction = parallel_prediction
        
        assert self.num_layer >= 1
                
        # NC GNN Message passing layers
        #   Note: The paper seems to not specify details about MLP layers
        #   Hence, we use basic 2-layer MLPS

        self.MLP1_ls, self.MLP2_ls, self.EPS_ls  = ModuleList([]), ModuleList([]), torch.nn.ParameterList([])
        self.dropout = Dropout(p=drop_ratio)
        for _ in range(self.num_layer):
            self.MLP1_ls.append(get_mlp(num_layers=2, 
                            in_dim=self.emb_dim, 
                            out_dim=self.emb_dim, 
                            hidden_dim=self.emb_dim // 2, 
                            activation=self.activation, 
                            dropout_rate=drop_ratio))
            self.MLP2_ls.append(get_mlp(num_layers=2, 
                            in_dim=self.emb_dim, 
                            out_dim=self.emb_dim, 
                            hidden_dim=self.emb_dim // 2, 
                            activation=self.activation, 
                            dropout_rate=drop_ratio))
            self.EPS_ls.append(torch.nn.Parameter(torch.tensor([0.0])))

        # Jumping Knowledge
        #   Note: The paper seems to not specify details about JK
        #   Here we use mean JK. However, they could have also used concat JK
        self.JK = lambda ls: torch.squeeze(torch.mean(torch.stack(ls), 0, True), 0)

        # Prediction Layer
        if prediction_type in [PredictionType.NODE_PREDICTION, PredictionType.GRAPH_PREDICTION, PredictionType.EDGE_PREDICTION, PredictionType.UNDIR_EDGE_PREDICTION]:
            print(f"Graph pooling function: {graph_pooling}")
            self.pool = get_pooling_fct(graph_pooling)

            if prediction_type in [PredictionType.EDGE_PREDICTION, PredictionType.UNDIR_EDGE_PREDICTION]:
                in_dim = 2*self.emb_dim 
            else:
                in_dim = self.emb_dim 

            self.mlp = get_mlp(num_layers=num_mlp_layers, 
                            in_dim=in_dim, 
                            out_dim=self.num_classes*self.num_tasks, 
                            hidden_dim=self.emb_dim // 2, 
                            activation=self.activation, 
                            dropout_rate=drop_ratio)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_attr = self.edge_encoder(edge_attr)
        
        # Each entry is the embedding of all nodes per message passing layers 
        h_list = [self.node_encoder(x)]


        for layer, (mlp1, mlp2, eps) in enumerate(zip(self.MLP1_ls, self.MLP2_ls, self.EPS_ls)):

            # (1) PREVIOUS EMBEDDING
            h_prev = (1+eps) * h_list[-1]

            # (2) (Classical) MESSAGE PASSING
            src, dst = edge_index  
            h_src = (h_list[-1])[src]  
            messages = torch.relu(h_src + edge_attr) 
            h_mp = torch.zeros_like(h_list[-1])
            h_mp.index_add_(0, dst, messages)


            # (3) TRIANGLE MESSAGE PASSING
            dst, src1, src2, triangle_edge_idx = batched_data.nc_mapping.T
            h_src1, h_src2 = (h_list[-1])[src1], (h_list[-1])[src2]
            h_edge = edge_attr[triangle_edge_idx]
            h_triangle_individual = mlp2(h_src1 + h_src2 + h_edge)
            h_triangle = torch.zeros_like(h_list[-1])
            h_triangle.index_add_(0, dst, h_triangle_individual)

            # (1) + (2) + (3)
            h = mlp1(h_prev + h_mp + h_triangle)
            h = self.dropout(h)

            if self.residual:
                h += h_list[layer]
            
            # No ReLU for last layer
            if layer != self.num_layer - 1:
                h = self.activation(h)

            h_list.append(h)
        
        h_node = self.JK(h_list)
       
        if self.prediction_type == PredictionType.NODE_EMBEDDING:
            return h_node
        
        elif self.prediction_type == PredictionType.NODE_PREDICTION:
            prediction = self.mlp(h_node)
            
        elif self.prediction_type == PredictionType.GRAPH_PREDICTION:
            h_graph = self.pool(h_node, batched_data.batch)
            prediction = self.mlp(h_graph)
            
        elif self.prediction_type == PredictionType.UNDIR_EDGE_PREDICTION:
            
            # Edges1 = (u,v)
            # Edges2 = (v, u)
            edges1 = edge_index[:, batched_data.edges_to_target[:, 0]]
            edges2 = edge_index[:, batched_data.edges_to_target[:, 1]]
            
            h_edge_endpoints1 = h_node[edges1]
            h_edge1 = torch.concat((h_edge_endpoints1[0,:,:], h_edge_endpoints1[1,:,:]), dim=1)
            
            h_edge_endpoints2 = h_node[edges2]
            h_edge2 = torch.concat((h_edge_endpoints2[0,:,:], h_edge_endpoints2[1,:,:]), dim=1)
            
            # We make separate prediction for each direction of an edge and mean these predictions
            if self.parallel_prediction:
                h_edge_pred = (self.mlp(h_edge1) + self.mlp(h_edge2)) / 2
                return h_edge_pred
            
            # We pool the embeddings of the two directions for each edge and compute the prediction on this pooled embedding
            else:
                return self.mlp(h_edge1 + h_edge2)
        
        else: # PredictionType.EDGE_PREDICTION
            h_edge_endpoints = h_node[batched_data.edge_index]
            h_edge = torch.concat((h_edge_endpoints[0,:,:], h_edge_endpoints[1,:,:]), dim=1)
            prediction = self.mlp(h_edge)

        # Reshape prediction to fit task
        if self.num_tasks == 1:
            prediction = prediction.view(-1, self.num_classes)
        else:
            prediction.view(-1, self.num_tasks, self.num_classes)
        return prediction
