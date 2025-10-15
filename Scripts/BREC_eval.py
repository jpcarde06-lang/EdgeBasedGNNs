from brec.dataset import BRECDataset
from brec.evaluator import evaluate

import torch
from torch_geometric.transforms import Compose

from Models.EBGNN import EBGNN
from Models.mpnn import MPNN
from Misc.utils import PredictionType
from Misc.edge2WL_trafo import EBGNNTransform
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.GSN_transform import GSN_transform

emb_dim = 16


##########################################
# 
# EB-GNN
#
##########################################


dataset = BRECDataset(transform = Compose([AddZeroEdgeAttr(edge_attr_size=emb_dim), EBGNNTransform()]))
model = EBGNN(num_classes = 2, num_tasks = 1, num_layer = 9, emb_dim = emb_dim, ff=False,
                 gnn_type="", residual = True, drop_ratio = 0, JK = "last", graph_pooling = "sum",
                 node_encoder = lambda x: x * torch.ones(x.shape[0], emb_dim, device=x.device), edge_encoder= lambda x: torch.ones(x.shape[0],emb_dim, device=x.device),
                 num_mlp_layers = 2, activation ="relu", prediction_type = PredictionType.GRAPH_PREDICTION, parallel_prediction=False)

model = model.to("cuda:0")

evaluate(
    dataset, model, device=torch.device("cuda:0"), log_path="log.txt", training_config=None
)

##########################################
# 
# MPNN + C3
#
##########################################

dataset = BRECDataset(transform = Compose([AddZeroEdgeAttr(edge_attr_size=64), GSN_transform(dim=3)]))

model = MPNN(num_classes = 2, num_tasks = 1, num_layer = 10, emb_dim = 64, 
                 gnn_type="gin", residual = False, drop_ratio = 0, JK = "last", graph_pooling = "sum",
                 node_encoder = lambda x: torch.ones(x.shape[0], 64, device=x.device), edge_encoder= lambda x: torch.ones(x.shape[0],64, device=x.device), 
                 num_mlp_layers = 2, activation ="relu", prediction_type = PredictionType.GRAPH_PREDICTION, parallel_prediction=False)


model = model.to("cuda:0")

evaluate(
    dataset, model, device=torch.device("cuda:0"), log_path="log.txt", training_config=None
)