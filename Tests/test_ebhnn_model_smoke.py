"""
Smoke test for the EBHNN registry wiring: EBGNN (unchanged model class) driven
by a StampEncoder edge_encoder + Linear(1, emb_dim) node_encoder over batched
EB-HNN active-pair graphs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Models.EBGNN import EBGNN
from Models.encoder import StampEncoder
from Misc.utils import PredictionType
from Tests.test_ebhnn_trafo import (
    _CompatEBHNNTransform,
    pendant_triangle_hyperedge_index,
)


def test_forward_backward_smoke():
    torch.manual_seed(0)

    hyperedge_index = pendant_triangle_hyperedge_index()
    transform = _CompatEBHNNTransform()

    data_list = [
        transform(
            Data(
                hyperedge_index=hyperedge_index.clone(),
                num_nodes=4,
                x=torch.ones(4, 1),
            )
        )
        for _ in range(2)
    ]
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    emb_dim = 16
    model = EBGNN(
        num_classes=2,
        num_tasks=1,
        num_layer=2,
        emb_dim=emb_dim,
        gnn_type="ebhnn",
        residual=0,
        ff=0,
        drop_ratio=0.0,
        JK="last",
        graph_pooling="sum",
        node_encoder=torch.nn.Linear(1, emb_dim),
        edge_encoder=StampEncoder(emb_dim, num_size_bins=8),
        num_mlp_layers=2,
        activation="relu",
        prediction_type=PredictionType.GRAPH_PREDICTION,
    )

    output = model(batch)

    # (a) forward output shape and finiteness
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()

    # (b) backward runs and the StampEncoder's first Linear gets a gradient
    labels = torch.tensor([0, 1])
    loss = F.cross_entropy(output, labels)
    loss.backward()

    stamp_first_linear = model.edge_encoder.encoder[0]
    assert stamp_first_linear.weight.grad is not None
    assert stamp_first_linear.weight.grad.abs().sum().item() > 0
