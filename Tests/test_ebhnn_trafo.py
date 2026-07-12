"""
Tests for EB-HNN's degeneration to EB-GNN, its stamp semantics, and batching.

See Misc/EBHNN_trafo.py and Misc/EBGNN_trafo.py for the transforms under test.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Misc.EBGNN_trafo import EBGNN_transform
from Misc.EBHNN_trafo import EBHNN_transform, EBHNNTransform

# Undirected edges of a triangle {0,1,2} plus a pendant edge (2,3).
PENDANT_TRIANGLE_UNDIRECTED = [(0, 1), (1, 2), (0, 2), (2, 3)]


def pendant_triangle_edge_index():
    """Directed edge_index (both directions per undirected edge)."""
    edges = []
    for u, v in PENDANT_TRIANGLE_UNDIRECTED:
        edges.append((u, v))
        edges.append((v, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def pendant_triangle_hyperedge_index():
    """2-uniform hyperedge_index: one size-2 hyperedge per undirected edge."""
    cols = []
    for e, (u, v) in enumerate(PENDANT_TRIANGLE_UNDIRECTED):
        cols.append((u, e))
        cols.append((v, e))
    return torch.tensor(cols, dtype=torch.long).t().contiguous()


def rows_as_set(tensor):
    return set(map(tuple, tensor.tolist()))


class _CompatEBHNNTransform(EBHNNTransform):
    """
    PyG-version compat shim: newer torch_geometric makes BaseTransform an ABC
    requiring `forward`, which EBHNNTransform (written for older PyG) doesn't
    define, so it can't even be instantiated. EBHNNTransform overrides
    `__call__` directly, so it takes precedence over BaseTransform's in the
    MRO and this stub `forward` is never actually invoked.
    """

    def forward(self, data):  # pragma: no cover - dead code, see class docstring
        raise NotImplementedError


def test_degeneration():
    edge_index = pendant_triangle_edge_index()
    hyperedge_index = pendant_triangle_hyperedge_index()

    wl_g, wr_g, wt_g, _ = EBGNN_transform(edge_index)
    active_pairs, wl_h, wr_h, wt_h, stamp, _ = EBHNN_transform(hyperedge_index)

    # (a) active-pair set == directed-edge set
    edges_g = rows_as_set(edge_index.t())
    pairs_h = rows_as_set(active_pairs.t())
    assert edges_g == pairs_h

    # phi: active-pair index -> graph edge index, built by matching (u, v) tuples
    graph_idx = {tuple(p): i for i, p in enumerate(edge_index.t().tolist())}
    phi = torch.tensor(
        [graph_idx[tuple(p)] for p in active_pairs.t().tolist()], dtype=torch.long
    )

    # (b) after mapping through phi, row sets of wl/wr/wt agree, order-independent
    for name, mapping_g, mapping_h in [
        ("wl", wl_g, wl_h),
        ("wr", wr_g, wr_h),
        ("wt", wt_g, wt_h),
    ]:
        set_g = rows_as_set(mapping_g)
        set_h = rows_as_set(phi[mapping_h])
        assert set_g == set_h, f"{name} mismatch: only_g={set_g - set_h}, only_h={set_h - set_g}"

    # (c) every stamp row is one-hot on bin 0 (size-2 hyperedges only)
    expected_row = torch.zeros(stamp.size(1))
    expected_row[0] = 1.0
    for row in stamp:
        assert torch.equal(row, expected_row)


def test_stamp_atomic_vs_assembled():
    # H1: one hyperedge {0, 1, 2}
    h1_incidence = [(0, 0), (1, 0), (2, 0)]
    # H2: three hyperedges {0,1}, {1,2}, {0,2}
    h2_incidence = [(0, 0), (1, 0), (1, 1), (2, 1), (0, 2), (2, 2)]

    h1_idx = torch.tensor(h1_incidence, dtype=torch.long).t().contiguous()
    h2_idx = torch.tensor(h2_incidence, dtype=torch.long).t().contiguous()

    pairs1, wl1, wr1, wt1, stamp1, _ = EBHNN_transform(h1_idx)
    pairs2, wl2, wr2, wt2, stamp2, _ = EBHNN_transform(h2_idx)

    # Both yield the same 6 active pairs, in the same order (index alignment is trivial here).
    assert pairs1.size(1) == 6
    assert pairs2.size(1) == 6
    assert torch.equal(pairs1, pairs2)

    for name, m1, m2 in [("wl", wl1, wl2), ("wr", wr1, wr2), ("wt", wt1, wt2)]:
        assert rows_as_set(m1) == rows_as_set(m2), f"{name} row sets differ"

    # Stamps differ: H1 one-hot on the size-3 bin (index 1); H2 one-hot on the size-2 bin (index 0).
    bins = stamp1.size(1)
    expected1 = torch.zeros(bins)
    expected1[1] = 1.0
    expected2 = torch.zeros(bins)
    expected2[0] = 1.0

    for row in stamp1:
        assert torch.equal(row, expected1)
    for row in stamp2:
        assert torch.equal(row, expected2)


def test_batching():
    hyperedge_index = pendant_triangle_hyperedge_index()
    transform = _CompatEBHNNTransform()

    data_list = [
        transform(Data(hyperedge_index=hyperedge_index.clone(), num_nodes=4))
        for _ in range(2)
    ]

    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    P = data_list[0].edge_index.size(1)
    assert P == 8

    expected_edge_batch = torch.cat(
        [torch.zeros(P, dtype=torch.long), torch.ones(P, dtype=torch.long)]
    )
    assert torch.equal(batch.edge_batch, expected_edge_batch)

    for name in ["wl_mapping", "wr_mapping", "wt_mapping"]:
        single = getattr(data_list[0], name)
        merged = getattr(batch, name)
        n0 = single.size(0)

        # First graph's rows are unchanged.
        assert torch.equal(merged[:n0], single)
        # Second graph's rows are offset by exactly P (the two copies are identical).
        assert torch.equal(merged[n0:], single + P)
        # All indices stay within the batch's total active-pair count.
        assert merged.max().item() < 2 * P
