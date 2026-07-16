from collections import defaultdict

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

DEFAULT_NUM_SIZE_BINS = 8
MIN_HYPEREDGE_SIZE = 2  # smallest hyperedge 


class FastHyperGraph(Data):

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


def build_comembership(hyperedge_index, max_hyperedge_size=None):

    hyperedge_members = defaultdict(set)
    hyperedges_of = defaultdict(set)
    for node, edge in hyperedge_index.t().tolist():
        node, edge = int(node), int(edge)
        hyperedge_members[edge].add(node)
        hyperedges_of[node].add(edge)

    members = {e: sorted(ns) for e, ns in hyperedge_members.items()}

    if max_hyperedge_size is not None:
        oversized = {e for e, ns in members.items() if len(ns) > max_hyperedge_size}
        if oversized:
            members = {e: ns for e, ns in members.items() if e not in oversized}
            hyperedges_of = defaultdict(set, {
                node: {e for e in edges if e not in oversized}
                for node, edges in hyperedges_of.items()
            })
        print(f"build_comembership: skipped {len(oversized)} hyperedge(s) "
              f"exceeding max_hyperedge_size={max_hyperedge_size}")

    neighbors = defaultdict(set)
    for ns in members.values():
        for u in ns:
            for v in ns:
                if u != v:
                    neighbors[u].add(v)

    return members, hyperedges_of, neighbors


def EBHNN_transform(hyperedge_index, num_size_bins=DEFAULT_NUM_SIZE_BINS, do_test=False,
                     max_hyperedge_size=None):

    device = hyperedge_index.device
    members, hyperedges_of, neighbors = build_comembership(
        hyperedge_index, max_hyperedge_size=max_hyperedge_size
    )

    # sorted by (u, v). Both (u, v)
    # and (v, u) exist because co-membership is symmetric.
    co_member_pairs = []
    pair_idx = {}
    for u in sorted(neighbors.keys()):
        for v in sorted(neighbors[u]):
            pair_idx[(u, v)] = len(co_member_pairs)
            co_member_pairs.append((u, v))

    # Common co-members: triangles[(u, v)] = N(u) & N(v). This is the beta-witness
    triangles = {}
    for (u, v) in co_member_pairs:
        common = neighbors[u] & neighbors[v]
        triangles[(u, v)] = sorted(common)
        if do_test:
            # Invariant: beta witnesses are exactly the co-membership intersection.
            assert set(triangles[(u, v)]) == (neighbors[u] & neighbors[v])

    wl_mapping = []  # alpha
    wr_mapping = []  # gamma
    wt_mapping = []  # beta
    for (u, v) in co_member_pairs:
        idx_uv = pair_idx[(u, v)]

        # alpha: pull from pairs (u, w) for every co-member w of u
        for w in sorted(neighbors[u]):
            wl_mapping.append([pair_idx[(u, w)], idx_uv])

        # gamma: pull from pairs (v, w) for every co-member w of v
        for w in sorted(neighbors[v]):
            wr_mapping.append([pair_idx[(v, w)], idx_uv])

        # beta: pull from pairs (u, w) and (v, w) for every common co-member w.
        for w in triangles[(u, v)]:
            wt_mapping.append([pair_idx[(u, w)], pair_idx[(v, w)], idx_uv])

    P = len(co_member_pairs)

    if P == 0:
        co_member_pairs_t = torch.zeros((2, 0), dtype=torch.long)
    else:
        co_member_pairs_t = torch.tensor(co_member_pairs, dtype=torch.long).t().contiguous()

    def to_tensor(mapping, width):
        if len(mapping) == 0:
            return torch.zeros((0, width), dtype=torch.long)
        return torch.tensor(mapping, dtype=torch.long)

    wl_tensor = to_tensor(wl_mapping, 2)
    wr_tensor = to_tensor(wr_mapping, 2)
    wt_tensor = to_tensor(wt_mapping, 3)

    # Size: for each pair (u, v), histogram the sizes of the
    # hyperedges that contain both u and v.
    co_member_size = torch.zeros((P, num_size_bins), dtype=torch.float)
    for (u, v), p in pair_idx.items():
        for e in (hyperedges_of[u] & hyperedges_of[v]):
            size = len(members[e])
            b = min(max(size - MIN_HYPEREDGE_SIZE, 0), num_size_bins - 1)
            co_member_size[p, b] += 1.0

    pair_batch = torch.zeros(P, dtype=torch.long)

    return (
        co_member_pairs_t.to(device),
        wl_tensor.to(device),
        wr_tensor.to(device),
        wt_tensor.to(device),
        co_member_size.to(device),
        pair_batch.to(device),
    )


class EBHNNTransform(BaseTransform):

    def __init__(self, num_size_bins=DEFAULT_NUM_SIZE_BINS, max_hyperedge_size=None):
        self.num_size_bins = num_size_bins
        self.max_hyperedge_size = max_hyperedge_size

    def __call__(self, data: Data):
        co_member_pairs, wl_tensor, wr_tensor, wt_tensor, co_member_size, pair_batch = EBHNN_transform(
            data.hyperedge_index, num_size_bins=self.num_size_bins,
            max_hyperedge_size=self.max_hyperedge_size,
        )

        kwargs = dict(
            y=data.y if hasattr(data, "y") else None,
            x=data.x if hasattr(data, "x") else None,
            edge_index=co_member_pairs,
            edge_attr=co_member_size,
            num_nodes=data.num_nodes,
            wl_mapping=wl_tensor,
            wr_mapping=wr_tensor,
            wt_mapping=wt_tensor,
            edge_batch=pair_batch,
        )

        if hasattr(data, "edges_to_target"):
            kwargs["edges_to_target"] = data.edges_to_target

        return FastHyperGraph(**kwargs)
