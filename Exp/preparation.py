import os
import csv

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset, LRGBDataset, QM9, MalNetTiny
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree, LocalDegreeProfile
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import torch.nn as nn

from Models.mpnn import MPNN, get_mp_layer
from Models.EBGNN import EBGNN
from Models.encoder import NodeEncoder, EdgeEncoder, VOCNodeEncoder, VOCEdgeEncoder
from Models.mlp import MLP
from Models.utils import get_activation
from Models.NCGNN import NCGNN, NCGNNTransform
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from Misc.cosine_scheduler import get_cosine_schedule_with_warmup
from Misc.select_only_one_target import SelectOnlyOneTarget
from Misc.change_QM9_units import ChangeQM9Units
from Misc.one_hot_encode_target import OneHotEncodeTarget
from Misc.weighted_cross_entropy_loss import weighted_cross_entropy
from Misc.utils import PredictionType
from Misc.EBGNN_trafo import EBGNNTransform
from Misc.GSN_transform import GSN_transform
from Misc.QM_dataset import QM_Dataset, QM_task_level

# ESAN
from Models.ESAN.preprocessing import policy2transform
from Models.ESAN.models import DSnetwork, DSSnetwork

def get_prediction_type(dataset_name):
    if dataset_name.lower() == "pascalvoc-sp":
        return PredictionType.NODE_PREDICTION
    elif "QMD" in dataset_name:
        return QM_task_level(dataset_name.split("-")[1])
    return PredictionType.GRAPH_PREDICTION

from torch_geometric.data import InMemoryDataset

class FilteredDataset(InMemoryDataset):
    def __init__(self, dataset):
        super().__init__()
        # Keep only graphs where 'edge_index' is not empty
        filtered_list = [
            data for data in dataset
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0
        ]
        self.data, self.slices = self.collate(filtered_list)

        # Copy important attributes
        self._num_classes = getattr(dataset, 'num_classes', None)
        self._num_node_features = getattr(dataset, 'num_node_features', None)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_node_features(self):
        return self._num_node_features



def get_transform(args):
    transforms = []
    dataset_name_lowercase = args.dataset.lower()
    
    if dataset_name_lowercase == "malnettiny":
        transforms += [OneHotEncodeTarget(5), LocalDegreeProfile(), AddZeroEdgeAttr(args.emb_dim)]

    if args.model in ["EBGNN"]:
        transforms.append(EBGNNTransform())
        # transforms.append(FastEdgeGNN())

    if args.model in ["NCGNN"]:
        transforms.append(NCGNNTransform())
    
    if args.model in ["DSS", "DS"]:
        esan_transform = policy2transform(args.policy, args.num_hops)
        transforms.append(esan_transform)
    
    if dataset_name_lowercase == "csl":
        transforms.append(OneHotEncodeTarget(10))
        transforms.append(OneHotDegree(5))
        
    if "GSN" in args.model:
        dim = int(args.model[3:])
        transforms.append(GSN_transform(dim=dim))
    
    if args.attach_cycles > 0:
        transforms.append(GSN_transform(dim=args.attach_cycles))
        
    if dataset_name_lowercase == "csl":
        # Pad features if necessary (needs to be done after adding additional features from other transformation)
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))

        
    # For dataset name QM9_i we only predict the i-th target value
    if "qm9" in dataset_name_lowercase and "_" in dataset_name_lowercase:
        target = int(dataset_name_lowercase.split("_")[1])
        assert target >= 0 and target <= 18
        transforms.append(ChangeQM9Units())
        transforms.append(SelectOnlyOneTarget(target))
         
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))

    return Compose(transforms)

def load_indices(dataset_name, config):
    all_idx = {}
    for section in ['train', 'val', 'test']:
        with open(os.path.join(config.SPLITS_PATH, dataset_name,  f"{section}.index"), 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx

def load_dataset(args, config):
    transform = get_transform(args)
    dataset_name = args.dataset.lower()

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    # ZINC
    if dataset_name in ["zinc", "zinc_full"]:
        subset = "full" not in dataset_name
        datasets = [ZINC(root=dir, subset=subset, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
        
    # OGB graph level tasks
    elif dataset_name in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]

    elif "qmd" in dataset_name:

        target = args.dataset.split("-")[1]
        dataset = QM_Dataset(root=dir, pre_transform=transform, target=target)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
        
    # Cyclic Skip Link dataset
    elif dataset_name == "csl":
        indices = load_indices("CSL", config)
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[indices["train"][args.split]], dataset[indices["val"][args.split]], dataset[indices["test"][args.split]]]
        
    # Long Rage Graph Benchmark datsets
    elif dataset_name == "peptides-func":
        datasets = [LRGBDataset(root=dir, name='Peptides-func', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif dataset_name == "peptides-struct":
        datasets = [LRGBDataset(root=dir, name='Peptides-struct', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif dataset_name == "pascalvoc-sp":
        datasets = [LRGBDataset(root=dir, name='PascalVOC-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    # elif dataset_name == "coco-sp":
    #     datasets = [LRGBDataset(root=dir, name='COCO-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif dataset_name == "pcqm-contact":
        datasets = [LRGBDataset(root=dir, name='PCQM-Contact', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
        
    elif "qm9" in dataset_name:
        # Based on https://github.com/chrsmrrs/SpeqNets
        
        dataset = QM9(root=dir, pre_transform=transform)
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        
        indices = load_indices("QM9", config)
        datasets = [dataset[indices["train"][0]], dataset[indices["val"][0]], dataset[indices["test"][0]]]
    elif "malnettiny" == dataset_name:
        datasets = [MalNetTiny(root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    else:
        raise NotImplementedError("Unknown dataset")
    
    # Filter out edge-less graphs
    datasets = [FilteredDataset(dataset) for dataset in datasets]
    
    for ds in datasets:
        ds.has_mean = "qm9" in args.dataset.lower()
        
        if ds.has_mean:
            ds.mean, ds.std = mean, std
    
    # This can probably be written more elegantly
    if args.model.lower() in ["dss", "ds"]:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims, edge_feature_dims = [], []
    model = args.model.lower()
    dataset_name = args.dataset.lower()

    # Load node and edge encoder
    if args.model.lower() in ["dss", "ds"] and args.policy == "ego_nets_plus":
        node_feature_dims += [2,2]
        
        
    if "GSN" in args.model:
        dim = int(args.model[3:])
        # The 20 here means that we never expect more than 20 cycles of every length that go through every node
        node_feature_dims += [20]*(dim-2)
        
    if args.attach_cycles > 0:
        node_feature_dims += [20]*(args.attach_cycles - 2)
        
    if not args.do_drop_feat and dataset_name != "csl":
        if "zinc" in dataset_name:
            node_feature_dims.append(28)
            edge_feature_dims.append(4)
        elif dataset_name in ["pcqm-contact", "peptides-struct", "peptides-func", "ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
            node_feature_dims += get_atom_feature_dims()
            edge_feature_dims += get_bond_feature_dims()
        elif "qmd" in dataset_name:
            node_feature_dims += get_atom_feature_dims()
            edge_feature_dims_temp = get_bond_feature_dims()
            edge_feature_dims_temp[0] = 13
            edge_feature_dims += edge_feature_dims_temp
        elif "qm9" in dataset_name:
            node_feature_dims += [2, 2, 2, 2, 2, 10, 1, 1, 1, 1, 5]
            edge_feature_dims += [2, 2, 2, 1]
         
        if dataset_name == "pascalvoc-sp":
            node_encoder, edge_encoder = VOCNodeEncoder(emb_dim = args.emb_dim), VOCEdgeEncoder(emb_dim = args.emb_dim)
        elif dataset_name == "malnettiny":
            edge_encoder = lambda x: x
            node_encoder = nn.Sequential(
                    nn.Linear(5, args.emb_dim//2),
                    get_activation(args.activation),
                    nn.Linear(args.emb_dim//2, args.emb_dim)
                )
        else:
            node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
            edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=edge_feature_dims)
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x
             

    # Load model
    if model in ["gin", "gcn", "gat"] or "gsn" in model:  
        return MPNN(num_classes, 
                    num_tasks, 
                    args.num_mp_layers, 
                    args.emb_dim, 
                    gnn_type = "gin" if "gsn" in model else model, 
                    drop_ratio = args.drop_out, 
                    JK = args.JK, 
                    graph_pooling = args.pooling, 
                    edge_encoder=edge_encoder, 
                    node_encoder=node_encoder, 
                    num_mlp_layers = args.num_mlp_layers, 
                    residual=args.use_residual, 
                    activation=args.activation,
                    prediction_type=get_prediction_type(args.dataset),
                    parallel_prediction=args.parallel_prediction)
    elif model == "ebgnn":  
        return EBGNN(num_classes, 
                    num_tasks, 
                    args.num_mp_layers, 
                    args.emb_dim, 
                    gnn_type = model, 
                    drop_ratio = args.drop_out, 
                    JK = args.JK, 
                    graph_pooling = args.pooling, 
                    edge_encoder=edge_encoder, 
                    node_encoder=node_encoder, 
                    num_mlp_layers = args.num_mlp_layers, 
                    residual=args.use_residual, 
                    ff=args.use_ff,
                    activation=args.activation,
                    prediction_type=get_prediction_type(args.dataset),
                    parallel_prediction=args.parallel_prediction,
                    use_dot_product = args.use_dot_product)
    elif model == "ncgnn":
        return NCGNN(num_classes, 
                    num_tasks, 
                    args.num_mp_layers, 
                    args.emb_dim, 
                    gnn_type = "gin" if "gsn" in model else model, 
                    drop_ratio = args.drop_out, 
                    JK = args.JK, 
                    graph_pooling = args.pooling, 
                    edge_encoder=edge_encoder, 
                    node_encoder=node_encoder, 
                    num_mlp_layers = args.num_mlp_layers, 
                    residual=args.use_residual, 
                    activation=args.activation,
                    prediction_type=get_prediction_type(args.dataset),
                    parallel_prediction=args.parallel_prediction)
    elif model == "mlp":
            return MLP(num_node_level_layers = args.num_n_layers,
                       num_graph_level_layers = args.num_g_layers,
                       node_encoder = node_encoder, 
                       emb_dim = args.emb_dim, 
                       num_classes = num_classes, 
                       num_tasks = num_tasks, 
                       dropout_rate = args.drop_out, 
                       graph_pooling = args.pooling, 
                       activation = args.activation,
                       prediction_type=get_prediction_type(args.dataset))
    elif model == "dss":
        assert get_prediction_type(args.dataset) == PredictionType.GRAPH_PREDICTION
        
        return DSSnetwork(num_layers = args.num_mp_layers, 
                          in_dim = args.emb_dim, 
                          emb_dim = args.emb_dim, 
                          num_tasks = num_tasks, 
                          feature_encoder = node_encoder,
                          edge_encoder = edge_encoder,
                          GNNConv = lambda x: get_mp_layer(emb_dim = x, 
                                                           activation=get_activation(args.activation), 
                                                           mp_type=args.mp))
    elif model == "ds":
        assert get_prediction_type(args.dataset) == PredictionType.GRAPH_PREDICTION
        subgraph_gnn = MPNN(None, 
                            None, 
                            num_layer = args.num_mp_layers, 
                            emb_dim = args.emb_dim, 
                            gnn_type = args.mp, 
                            drop_ratio = args.drop_out, 
                            JK = "last", 
                            graph_pooling = None, 
                            edge_encoder=edge_encoder, 
                            node_encoder=node_encoder, 
                            num_mlp_layers = None, 
                            residual=args.use_residual, 
                            activation=args.activation,
                            prediction_type=PredictionType.NODE_EMBEDDING)
        
        channels = list(map(lambda x: int(x), args.channels.split('-')))
        return DSnetwork(subgraph_gnn=subgraph_gnn, 
                         subgraph_gnn_out_dim=args.emb_dim, 
                         channels=channels, num_tasks=num_tasks, 
                         invariant=args.use_invariant, 
                         subgraph_pool= args.pooling)
    else: 
        raise ValueError("Unknown model name")

    return model

def get_optimizer_scheduler(model, args):
    lr = args.lr
    scheduler_name = args.scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)

    match scheduler_name:
        case 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        args.scheduler_decay_steps,
                                                        gamma=args.scheduler_decay_rate)
        case 'None':
            scheduler = None
        case "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='min',
                                                                    factor=args.scheduler_decay_rate,
                                                                    patience=args.scheduler_patience,
                                                                    verbose=True)
        case "Cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps = args.warmup_steps, 
                                                        num_training_steps = args.epochs)
        case _:
            raise NotImplementedError(f'Scheduler {scheduler_name} is not currently supported.')

    return optimizer, scheduler

def get_loss(dataset_name):
    metric_method = None
    dataset_name_lowercase = dataset_name.lower()
    if dataset_name_lowercase in ["peptides-struct", "zinc", "zinc_full"] or "qm9" in dataset_name_lowercase or "qmd" in dataset_name_lowercase:
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif dataset_name_lowercase in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase in ["csl", "malnettiny"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif dataset_name_lowercase in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name == "peptides-func":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap" 
    elif dataset_name_lowercase == "pascalvoc-sp":
        loss = weighted_cross_entropy
        metric = "f1" 
        metric_method = F1Score(task="multiclass", num_classes=21, average='macro')
    elif dataset_name_lowercase == "pcqm-contact":
        loss = torch.nn.CrossEntropyLoss()
        metric = "mrr"
        metric_method = mrr_fct
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def mrr_fct(y_pred_pos, y_pred_neg):
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked first among those with equal score
    # optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    
    # Need to unroll it because I do not have enough RAM otherwise    
    optimistic_rank = []
    pessimistic_rank = []
    
    for i, a in enumerate(y_pred_pos.view(-1, 1)):
        optimistic_rank.append(sum(y_pred_neg > a))
        pessimistic_rank.append(sum(y_pred_neg >= a))
        
    optimistic_rank = torch.Tensor(optimistic_rank)
    pessimistic_rank = torch.Tensor(pessimistic_rank) 
    
    # pessimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked last among those with equal score
    # pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)
    return float(mrr_list.mean().item())


def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method