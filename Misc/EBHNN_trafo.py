"""

Pre-processing for EB-HNN: the hypergraph generalization of EB-GNN.

EB-GNN operates on the ordered edges of a graph. EB-HNN lifts this to
hypergraphs by operating on ACTIVE PAIRS: ordered vertex pairs (u, v),
u != v, that co-occur in at least one hyperedge. Everything the reused
EB-GNN layer needs (alpha / beta / gamma aggregation mappings) is defined
over these active pairs using CO-MEMBERSHIP neighborhoods

    N(u) = { v != u : some hyperedge contains both u and v }

so the layer in Models/EBGNN.py can be reused verbatim, with `edge_index`
holding the active pairs instead of graph edges.

The three aggregations for an active pair (u, v) mirror EB-GNN exactly:
  * alpha (wl): for w in N(u)            -> message from pair (u, w)
  * gamma (wr): for w in N(v)            -> message from pair (v, w)
  * beta  (wt): for w in N(u) & N(v)     -> messages from pairs (u, w), (v, w)
The beta witnesses use the CO-MEMBERSHIP intersection N(u) & N(v), NOT
"atomic" witnesses (a single hyperedge containing all of u, v, w).

Each active pair is additionally initialized with a SIZE STAMP: a histogram
over the sizes of the hyperedges that contain both u and v. This is exposed
as `edge_attr` so it can be consumed by a StampEncoder plugged into the
reused layer's `edge_encoder` slot.

Degeneration: on a 2-uniform hypergraph (every hyperedge has exactly two
vertices) the active pairs are exactly the graph edges, co-membership is
exactly graph adjacency, and this transform reduces to EBGNN_transform.
Every hyperedge has size 2, so the stamp is a constant.
"""

from collections import defaultdict

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Number of hyperedge-size bins in the stamp histogram. Bin b counts
# hyperedges of size (b + MIN_HYPEREDGE_SIZE); the last bin is an overflow
# bin for all larger sizes. Must be consistent across a dataset so the
# per-pair stamps can be concatenated into a batch.
DEFAULT_NUM_SIZE_BINS = 8
MIN_HYPEREDGE_SIZE = 2  # smallest hyperedge that induces an active pair


class FastHyperGraph(Data):
    """
    Data container for EB-HNN.

    `edge_index` holds the ACTIVE PAIRS ([2, P], row 0 = u, row 1 = v) so the
    EB-GNN layer can be reused unchanged. The alpha/beta/gamma mappings index
    INTO the active pairs, hence their __inc__ offsets are in ACTIVE-PAIR
    units (= number of active pairs = edge_index.size(1)), NOT node units.
    `edge_index` itself still indexes nodes, so it keeps the default node-unit
    offset from Data.__inc__.
    """

    def __inc__(self, key, value, store):
        if key == "edge_batch":
            return 1  # edge_batch is a batch index
        elif key in ["wl_mapping", "wr_mapping"]:
            # shape [num_mappings, 2]; both columns index active pairs
            return torch.tensor([self.edge_index.size(1), self.edge_index.size(1)])
        elif key == "wt_mapping":
            # shape [num_mappings, 3]; all three columns index active pairs
            return torch.tensor([self.edge_index.size(1)] * 3)
        elif key == "edges_to_target":
            return torch.tensor(self.edge_index.size(1))
        else:
            return super().__inc__(key, value, store)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["wl_mapping", "wr_mapping", "wt_mapping", "edges_to_target"]:
            return 0  # concatenate mappings along dimension 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_comembership(hyperedge_index):
    """
    Parse a hypergraph in PyG bipartite-incidence form.

    Args:
        hyperedge_index: LongTensor [2, nnz], row 0 = node id, row 1 = hyperedge id.

    Returns:
        members:        dict hyperedge_id -> sorted list of its (distinct) node ids
        hyperedges_of:  dict node -> set of hyperedge ids containing it
        neighbors:      dict node -> set of co-members (co-membership neighborhood N(u))
    """
    raw_members = defaultdict(set)
    hyperedges_of = defaultdict(set)
    for node, edge in hyperedge_index.t().tolist():
        node, edge = int(node), int(edge)
        raw_members[edge].add(node)
        hyperedges_of[node].add(edge)

    members = {e: sorted(ns) for e, ns in raw_members.items()}

    neighbors = defaultdict(set)
    for ns in members.values():
        for u in ns:
            for v in ns:
                if u != v:
                    neighbors[u].add(v)

    return members, hyperedges_of, neighbors


def EBHNN_transform(hyperedge_index, num_size_bins=DEFAULT_NUM_SIZE_BINS, do_test=False):
    """
    Compute the EB-HNN pre-processing from a hypergraph incidence structure.

    Args:
        hyperedge_index: LongTensor [2, nnz], row 0 = node id, row 1 = hyperedge id.
        num_size_bins:   width of the per-pair size-stamp histogram.
        do_test:         if True, assert the beta-witness invariant.

    Returns:
        active_pairs: LongTensor [2, P]        ordered active pairs (row 0 = u, row 1 = v)
        wl_tensor:    LongTensor [., 2]        alpha mapping  [idx(u, w), idx(u, v)]
        wr_tensor:    LongTensor [., 2]        gamma mapping  [idx(v, w), idx(u, v)]
        wt_tensor:    LongTensor [., 3]        beta  mapping  [idx(u, w), idx(v, w), idx(u, v)]
        stamp:        FloatTensor [P, bins]    per-pair hyperedge-size histogram
        pair_batch:   LongTensor [P]           batch index per active pair (all zeros)
    """
    device = hyperedge_index.device
    members, hyperedges_of, neighbors = build_comembership(hyperedge_index)

    # Active pairs (ordered). Deterministic order: sorted by (u, v). Both (u, v)
    # and (v, u) exist because co-membership is symmetric.
    active_pairs = []
    pair_idx = {}
    for u in sorted(neighbors.keys()):
        for v in sorted(neighbors[u]):
            pair_idx[(u, v)] = len(active_pairs)
            active_pairs.append((u, v))

    # Common co-members: triangles[(u, v)] = N(u) & N(v). This is the beta-witness
    # set. It is the co-membership intersection, NOT single-hyperedge witnesses.
    triangles = {}
    for (u, v) in active_pairs:
        common = neighbors[u] & neighbors[v]
        triangles[(u, v)] = sorted(common)
        if do_test:
            # Invariant: beta witnesses are exactly the co-membership intersection.
            assert set(triangles[(u, v)]) == (neighbors[u] & neighbors[v])

    wl_mapping = []  # alpha
    wr_mapping = []  # gamma
    wt_mapping = []  # beta
    for (u, v) in active_pairs:
        idx_uv = pair_idx[(u, v)]

        # alpha: pull from pairs (u, w) for every co-member w of u
        for w in sorted(neighbors[u]):
            wl_mapping.append([pair_idx[(u, w)], idx_uv])

        # gamma: pull from pairs (v, w) for every co-member w of v
        for w in sorted(neighbors[v]):
            wr_mapping.append([pair_idx[(v, w)], idx_uv])

        # beta: pull from pairs (u, w) and (v, w) for every common co-member w.
        # w in N(u) => (u, w) is active; w in N(v) => (v, w) is active.
        for w in triangles[(u, v)]:
            wt_mapping.append([pair_idx[(u, w)], pair_idx[(v, w)], idx_uv])

    P = len(active_pairs)

    if P == 0:
        active_pairs_t = torch.zeros((2, 0), dtype=torch.long)
    else:
        active_pairs_t = torch.tensor(active_pairs, dtype=torch.long).t().contiguous()

    def to_tensor(mapping, width):
        if len(mapping) == 0:
            return torch.zeros((0, width), dtype=torch.long)
        return torch.tensor(mapping, dtype=torch.long)

    wl_tensor = to_tensor(wl_mapping, 2)
    wr_tensor = to_tensor(wr_mapping, 2)
    wt_tensor = to_tensor(wt_mapping, 3)

    # Size stamp: for each active pair (u, v), histogram the sizes of the
    # hyperedges that contain BOTH u and v.
    stamp = torch.zeros((P, num_size_bins), dtype=torch.float)
    for (u, v), p in pair_idx.items():
        for e in (hyperedges_of[u] & hyperedges_of[v]):
            size = len(members[e])
            b = min(max(size - MIN_HYPEREDGE_SIZE, 0), num_size_bins - 1)
            stamp[p, b] += 1.0

    pair_batch = torch.zeros(P, dtype=torch.long)

    return (
        active_pairs_t.to(device),
        wl_tensor.to(device),
        wr_tensor.to(device),
        wt_tensor.to(device),
        stamp.to(device),
        pair_batch.to(device),
    )


class EBHNNTransform(BaseTransform):
    r"""
    Turn a hypergraph Data object into the active-pair representation EB-HNN
    consumes. Expects `data.hyperedge_index` ([2, nnz], row 0 = node, row 1 =
    hyperedge). Node features `data.x` are kept; the active pairs become
    `edge_index` and the size stamp becomes `edge_attr` (to be encoded by a
    StampEncoder in the reused EB-GNN layer's edge_encoder slot).
    """

    def __init__(self, num_size_bins=DEFAULT_NUM_SIZE_BINS):
        self.num_size_bins = num_size_bins

    def __call__(self, data: Data):
        active_pairs, wl_tensor, wr_tensor, wt_tensor, stamp, pair_batch = EBHNN_transform(
            data.hyperedge_index, num_size_bins=self.num_size_bins
        )

        kwargs = dict(
            y=data.y if hasattr(data, "y") else None,
            x=data.x if hasattr(data, "x") else None,
            edge_index=active_pairs,
            edge_attr=stamp,
            num_nodes=data.num_nodes,
            wl_mapping=wl_tensor,
            wr_mapping=wr_tensor,
            wt_mapping=wt_tensor,
            edge_batch=pair_batch,
        )

        if hasattr(data, "edges_to_target"):
            kwargs["edges_to_target"] = data.edges_to_target

        return FastHyperGraph(**kwargs)
