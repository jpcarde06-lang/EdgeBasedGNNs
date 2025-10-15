"""
Speed evaluation on QM9
"""

from Exp.preparation import load_dataset, get_transform, get_model, get_optimizer_scheduler, get_loss, get_prediction_type
from Exp.parser import parse_args
from Exp.training_loop_functions import train

from torch_geometric.datasets import QM9

from Misc.config import config

import time, numpy as np

repeats = 3

#######################################
#
# PREPREOCESSING EVAL
#
#######################################



ds_parameters = parse_args({
        "--dataset": "QM9_0",
        "--batch_size": 1
        })

transform = get_transform(parse_args({
        "--dataset": "ZINC",
        "--model": "EBGNN"
        }))

times = []
for idx in range(repeats):
    print(idx)

    ds = load_dataset(ds_parameters, config)
    transformed_graphs = []
    start = time.perf_counter()
    for split in ds:
        for graph in split:
            transformed_graphs.append(transform(graph))

    end = time.perf_counter()
    print(end-start)
    times.append(end - start)

times = np.array(times)
print("Pre processing")
print(f"mean={times.mean():.4f}s, std={times.std():.4f}s")



#######################################
#
# TRAINING EVAL
#
#######################################

training_parameters = parse_args({
        "--dataset": "QM9_0",
        "--batch_size": 64,
        "--model": "EBGNN",
        "--emb_dim": 256,
        "--ff": 1,
        "--num_mp_layers": 5,
        "--pooling": "nodesum",
        "--residual": 1,
        })

train_loader, val_loader, test_loader =  load_dataset(training_parameters, config)
device = training_parameters.device
model = get_model(training_parameters, 1, train_loader.dataset.num_node_features, 1).to(device)
optimizer, scheduler = get_optimizer_scheduler(model, training_parameters)
loss_dict = get_loss("QM9_0")
loss_fct = loss_dict["loss"]
eval_name = loss_dict["metric"]
metric_method = loss_dict["metric_method"]
prediction_type = get_prediction_type("qm9_0")

times = []
for idx in range(repeats):
    start = time.perf_counter()
    train_result = train(model, device, train_loader, optimizer, loss_fct, eval_name, None, metric_method=metric_method, prediction_type=prediction_type)
    end = time.perf_counter()
    print(end-start)
    times.append(end - start)

times = np.array(times)
print("Training")
print(f"mean={times.mean():.4f}s, std={times.std():.4f}s")