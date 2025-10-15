"""

EB-GNN model.

"""

import copy

import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d, Dropout
import torch.nn as nn
from torch_scatter import scatter, scatter_add, scatter_mean

from Models.utils import get_pooling_fct, get_activation
from Misc.utils import PredictionType

class FlexibleBatchNorm1d(nn.Module):
    """
    BatchnNorms and LinearLayers have incompatible shapes (for conditioning). 
    This wraps around batchNorms to make them compatible
    """
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.dim() == 2:
            # Standard case: [B, F]
            return self.bn(x)
        elif x.dim() == 3:
            B, T, F = x.shape

            # Permute to [B, F, T] so feature dim is second
            x_perm = x.permute(0, 2, 1)  # [num_nodes, emb_dim+1, num_targets]

            # Apply batchnorm directly (3D input with feature dim second)
            x_bn = self.bn(x_perm)

            # Permute back to original: [num_nodes, num_targets, emb_dim + 1]
            x_out = x_bn.permute(0, 2, 1)

            return x_out
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")
        
    def reset_parameters(self):
        self.bn.reset_parameters()

class EBGNN(torch.nn.Module):
    def __init__(self, num_classes, num_tasks, num_layer, emb_dim, 
                 gnn_type, residual, ff, drop_ratio, JK, graph_pooling,
                 node_encoder, edge_encoder, num_mlp_layers, activation, prediction_type, edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0, 
                 parallel_prediction = True,
                 use_dot_product = False):
        
        super(EBGNN, self).__init__()
        
        self.parallel_prediction = parallel_prediction
        
        print(f"parallel_prediction: {self.parallel_prediction}")
        self.graph_pooling = graph_pooling
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.residual = residual
        self.ff = ff
        self.node_encoder1 = node_encoder
        self.node_encoder2 = copy.deepcopy(node_encoder)
        self.use_dot_product = use_dot_product

        self.edge_encoder = edge_encoder
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.activation = get_activation(activation)
        self.prediction_type = prediction_type
                
        assert self.num_layer >= 1
        
        # Simple input normalization
        self.input_norm = FlexibleBatchNorm1d(emb_dim)

        if JK == "special":
            self.register_parameter("jkparams", torch.nn.Parameter(torch.randn((num_layer,))))

        def build_mlp():
            return Sequential(
                Linear(emb_dim, emb_dim),
                FlexibleBatchNorm1d(emb_dim),
                self.activation            )
        
        def build_w2_mlp():
            return Sequential(
                Linear(2 * emb_dim, emb_dim),
                FlexibleBatchNorm1d(emb_dim),
                self.activation
            )

        self.w1_mlps = ModuleList([build_mlp() for _ in range(num_layer)])

        
        if self.use_dot_product:
            self.w2_mlps, self.w2_mlps2 = ModuleList([build_mlp() for _ in range(num_layer)]), ModuleList([build_mlp() for _ in range(num_layer)])
            self.w2_bn = ModuleList([FlexibleBatchNorm1d(emb_dim) for _ in range(num_layer)])
        else:
            self.w2_mlps = ModuleList([build_w2_mlp() for _ in range(num_layer)])
            self.w2_mlps2 = None

        self.w3_mlps = ModuleList([build_mlp() for _ in range(num_layer)])
        self.w4_mlps = ModuleList([build_mlp() for _ in range(num_layer)])
        
        if self.ff:
            self.ffn_layers = ModuleList([
                Sequential(
                    Linear(emb_dim, 2 * emb_dim),
                    self.activation,
                    Linear(2 * emb_dim, emb_dim)
                ) for _ in range(num_layer)
            ])
        
        self.node_mapping_src = Linear(emb_dim, emb_dim)
        self.node_mapping_dst = Linear(emb_dim, emb_dim)   
        
        self.node_dropout = torch.nn.Dropout(xdropout)

        self.layer_norms = ModuleList([FlexibleBatchNorm1d(emb_dim) for _ in range(num_layer)])
        self.layer_norms2 = ModuleList([FlexibleBatchNorm1d(emb_dim) for _ in range(num_layer)])

        self.dropouts = ModuleList([Dropout(p=drop_ratio) for _ in range(num_layer)])
        
        if prediction_type in [PredictionType.GRAPH_PREDICTION, PredictionType.EDGE_PREDICTION, PredictionType.UNDIR_EDGE_PREDICTION]:
            self.pool = get_pooling_fct(graph_pooling)
            self.graph_pred_mlp = Sequential(
                Linear(emb_dim, emb_dim // 2),
                self.activation,
                Dropout(p=drop_ratio),
                Linear(emb_dim // 2, self.num_classes * self.num_tasks),
            )
            
        if prediction_type == PredictionType.UNDIR_EDGE_PREDICTION:
            self.final_bn = FlexibleBatchNorm1d(emb_dim)
            
    def reset_parameters(self):
        # Reset individual modules
        self.input_norm.reset_parameters()
        self.node_mapping_src.reset_parameters()
        self.node_mapping_dst.reset_parameters()
       
        # Reset module lists
        for module_list in [self.w1_mlps, self.w2_mlps, self.w3_mlps, self.w4_mlps, self.layer_norms, self.dropouts]:
            for module in module_list:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        
        # Reset prediction head if exists
        if hasattr(self, 'graph_pred_mlp'):
            for module in self.graph_pred_mlp:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        
    def scatter_with_mlp(self, h, mapping, mlp, mlp2 = None):
        if mapping.size(0) == 0:
            return 0
        
        # beta aggregation
        if mapping.shape[1] == 3:  
            src1, src2, dst = mapping[:, 0], mapping[:, 1], mapping[:, 2]
            if mlp2 is None:
                features = torch.cat([h[src1], h[src2]], dim=-1)
                return scatter(mlp(features), dst, dim=0, reduce="sum", dim_size=h.shape[0])
            else:
                # Product combination
                features = torch.tanh(mlp(h[src1]) * mlp2(h[src2]))
                return scatter(features, dst, dim=0, reduce="sum", dim_size=h.shape[0])

        # alpha and gamma aggregation
        else:  
            src, dst = mapping[:, 0], mapping[:, 1]
            features = h[src] 
            return scatter(mlp(features), dst, dim=0, reduce="sum", dim_size=h.shape[0])
        
    def forward(self, batched_data):
        x, batch = batched_data.x, batched_data.batch
    
        h_node1 = x[batched_data.edge_index[0]]
        h_node2 = x[batched_data.edge_index[1]]        
        h_node = self.node_encoder1(h_node1) + self.node_encoder2(h_node2)
        
        if batched_data.edge_attr is None:
            h = h_node
        else: 
            edge_attr = batched_data.edge_attr
            h_edge = self.edge_encoder(edge_attr)
            h = h_node + h_edge
        
        h_list = [h]
        
        for layer in range(self.num_layer):
            h_prev = h

            wl = self.scatter_with_mlp(h, batched_data.wl_mapping, self.w1_mlps[layer])
            wt = self.scatter_with_mlp(h, batched_data.wt_mapping, self.w2_mlps[layer], None if self.w2_mlps2 is None else self.w2_mlps2[layer])
            wr = self.scatter_with_mlp(h, batched_data.wr_mapping, self.w3_mlps[layer])

            # Combine everything
            h = wl + wt + wr
            
            h = self.activation(h)
            
            if self.residual:
                h = h + h_prev
            
            if self.ff:
                h = self.layer_norms[layer](h)
                h = h + self.ffn_layers[layer](h)
            
            h = self.layer_norms2[layer](h)
            h = self.dropouts[layer](h)
            h_list.append(h)
        
        if self.JK == "last":
            h_final = h_list[-1]
        elif self.JK == "sum":
            h_final = sum(h_list)
        elif self.JK == "max":
            h_final = torch.stack(h_list, dim=0).max(dim=0)[0]#
        elif self.JK == "special":
            jkx = torch.stack(h_list, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            h_final = torch.sum(jkx*sftmax, dim=0)
        else:
            h_final = h_list[-1]
                
        if self.prediction_type == PredictionType.GRAPH_PREDICTION:
            batch_index = batched_data.edge_batch if hasattr(batched_data, "edge_batch") else batch
            
            if self.graph_pooling ==  "nodesum":
                h_graph = self.pool(h_final, batch_index, batched_data.batch)
            else:
                h_graph = self.pool(h_final, batch_index)
            prediction = self.graph_pred_mlp(h_graph)
            
            if self.num_tasks == 1:
                prediction = prediction.view(-1, self.num_classes)
            else:
                prediction = prediction.view(-1, self.num_tasks, self.num_classes)
            return prediction
        elif self.prediction_type  == PredictionType.EDGE_EMBEDDING:
            return  h_final
        elif self.prediction_type == PredictionType.NODE_EMBEDDING:
            src, dst = batched_data.edge_index
            node_emb = torch.zeros(x.shape[0], self.emb_dim, dtype=h_final.dtype, device=h_final.device)
            scatter_add(h_final / 2, src, dim=0, out=node_emb)
            scatter_add(h_final / 2, dst, dim=0, out=node_emb)
            
            return node_emb

        elif self.prediction_type == PredictionType.EDGE_PREDICTION:
            return self.graph_pred_mlp(h_final)
        
        elif self.prediction_type == PredictionType.UNDIR_EDGE_PREDICTION:
            
            # We make separate prediction for each direction of an edge and mean these predictions
            if self.parallel_prediction:
                h_edge_pred = torch.sum(self.graph_pred_mlp(h_final)[batched_data.edges_to_target], dim=1) / 2
                return h_edge_pred
            
            # We pool the embeddings of the two directions for each edge and compute the prediction on this pooled embedding
            else:
                h_edge_pred = self.final_bn(torch.sum(h_final[batched_data.edges_to_target], dim=1))
                return self.graph_pred_mlp(h_edge_pred)

        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")