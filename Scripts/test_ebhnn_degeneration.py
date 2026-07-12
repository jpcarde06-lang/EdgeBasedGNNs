"""
Degeneration test for EB-HNN.

On a 2-uniform hypergraph (every hyperedge has exactly two vertices) EB-HNN
must reduce to EB-GNN. We verify this by decoding the alpha/beta/gamma mapping
rows of both transforms into node-space tuples and checking the SETS are
identical. Decoding to node-space makes the comparison independent of the
(arbitrary) active-pair / edge indexing order, which is the correct notion of
equivalence because the layer's scatter-sum aggregation is permutation
invariant over mapping rows.

We also sanity-check a genuinely non-2-uniform hypergraph: the beta-witness
invariant triangles[(u, v)] == N(u) & N(v) (via do_test).

Run:  PYTHONPATH=. python Scripts/test_ebhnn_degeneration.py
"""

import torch

from Misc.EBGNN_trafo import EBGNN_transform
from Misc.EBHNN_trafo import EBHNN_transform


def random_undirected_edge_index(n, p, gen):
    """Erdos-Renyi undirected graph as an edge_index with BOTH directions."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if torch.rand((), generator=gen).item() < p:
                edges.append((u, v))
                edges.append((v, u))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def two_uniform_hypergraph_from_edge_index(edge_index):
    """
    Build the 2-uniform hyperedge_index equivalent to an undirected graph:
    one hyperedge per undirected edge {u, v}.
    Returns hyperedge_index [2, nnz], row 0 = node, row 1 = hyperedge id.
    """
    undirected = set()
    for u, v in edge_index.t().tolist():
        undirected.add((min(u, v), max(u, v)))

    cols = []
    for e, (u, v) in enumerate(sorted(undirected)):
        cols.append((u, e))
        cols.append((v, e))
    if not cols:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(cols, dtype=torch.long).t().contiguous()


def decode_mapping(mapping, index):
    """
    Decode a mapping tensor [., k] of token indices into a set of tuples of
    node-pairs, using `index` ([2, T]) to map token -> (u, v).
    """
    idx = index.t().tolist()  # token -> [u, v]
    out = set()
    for row in mapping.tolist():
        out.add(tuple((idx[t][0], idx[t][1]) for t in row))
    return out


def check_graph(edge_index, label):
    hyperedge_index = two_uniform_hypergraph_from_edge_index(edge_index)

    wl_g, wr_g, wt_g, _ = EBGNN_transform(edge_index, do_test=False)
    pairs, wl_h, wr_h, wt_h, stamp, pair_batch = EBHNN_transform(
        hyperedge_index, do_test=True
    )

    for name, mg, mh in [
        ("alpha/wl", wl_g, wl_h),
        ("gamma/wr", wr_g, wr_h),
        ("beta/wt", wt_g, wt_h),
    ]:
        set_g = decode_mapping(mg, edge_index)
        set_h = decode_mapping(mh, pairs)
        assert set_g == set_h, (
            f"[{label}] {name} mismatch\n"
            f"  only in EBGNN: {sorted(set_g - set_h)[:5]}\n"
            f"  only in EBHNN: {sorted(set_h - set_g)[:5]}"
        )

    # Active pairs must equal the graph edges (as a set of ordered pairs).
    edges_g = set(map(tuple, edge_index.t().tolist()))
    edges_h = set(map(tuple, pairs.t().tolist()))
    assert edges_g == edges_h, f"[{label}] active pairs != graph edges"

    # On a 2-uniform hypergraph every common hyperedge has size 2 -> the stamp
    # is entirely in bin 0 (size 2) and constant across active pairs.
    if stamp.numel() > 0:
        assert stamp[:, 1:].sum().item() == 0, f"[{label}] stamp has non-size-2 mass"
        assert (stamp[:, 0] == stamp[0, 0]).all(), f"[{label}] stamp not constant"

    print(f"[{label}] OK  ({pairs.size(1)} active pairs, "
          f"wl={wl_h.size(0)}, wr={wr_h.size(0)}, wt={wt_h.size(0)})")


def check_non_uniform_invariant():
    """Non-2-uniform hypergraph: exercise the beta-witness invariant (do_test)."""
    # hyperedges: {0,1,2} (size 3), {2,3} (size 2), {0,3,4} (size 3)
    incidence = [
        (0, 0), (1, 0), (2, 0),
        (2, 1), (3, 1),
        (0, 2), (3, 2), (4, 2),
    ]
    hyperedge_index = torch.tensor(incidence, dtype=torch.long).t().contiguous()
    pairs, wl, wr, wt, stamp, _ = EBHNN_transform(hyperedge_index, do_test=True)
    # Spot-check a known co-membership neighborhood: node 0 co-occurs with
    # {1,2} (hyperedge 0) and {3,4} (hyperedge 2) -> N(0) = {1,2,3,4}.
    idx = pairs.t().tolist()
    n0 = {v for (u, v) in idx if u == 0}
    assert n0 == {1, 2, 3, 4}, f"N(0) wrong: {n0}"
    # Stamp of pair (0,1): only hyperedge 0 (size 3) contains both -> bin 1.
    pmap = {tuple(p): i for i, p in enumerate(idx)}
    s01 = stamp[pmap[(0, 1)]]
    assert s01[1].item() == 1.0 and s01.sum().item() == 1.0, f"stamp(0,1)={s01.tolist()}"
    print(f"[non-uniform] OK  N(0)={sorted(n0)}, stamp(0,1)={s01.tolist()}")


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)

    # Deterministic small hand cases.
    triangle = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    check_graph(triangle, "triangle")

    path = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    check_graph(path, "path-0-1-2-3")

    # Random Erdos-Renyi graphs.
    for seed in range(12):
        g = torch.Generator().manual_seed(seed)
        n = 8 + (seed % 5)
        ei = random_undirected_edge_index(n, 0.4, g)
        check_graph(ei, f"ER(n={n},seed={seed})")

    check_non_uniform_invariant()
    print("\nAll degeneration + invariant checks passed.")
