"""
Loader for the k-GWL graph-classification hypergraph benchmarks (STEAM,
TWITTER, IMDB variants).

File format (see Data/data/hypergraph/<DATASET>/*.txt):
    line 1: N = number of hypergraphs
    per hypergraph:
        header line: "num_v num_e label"
        one line of num_v vertex-label tokens (unused for features; only
        parsed to advance the reader)
        num_e lines, each the space-separated vertex ids of one hyperedge
"""

import os

import torch
from torch_geometric.data import Data, InMemoryDataset

NAME_TO_FILE = {
    "steam_player": os.path.join("STEAM", "steam_player.txt"),
    "twitter_friend": os.path.join("TWITTER", "twitter_friend.txt"),
    "IMDB_dir_form": os.path.join("IMDB", "IMDB_dir_form.txt"),
    "IMDB_wri_form": os.path.join("IMDB", "IMDB_wri_form.txt"),
    "IMDB_dir_genre": os.path.join("IMDB", "IMDB_dir_genre.txt"),
    "IMDB_wri_genre": os.path.join("IMDB", "IMDB_wri_genre.txt"),
}


def _parse_kgwl_file(path):
    """
    Parse a k-GWL txt file into a list of raw hypergraph records.

    Returns a list of dicts with keys:
        num_v, num_e, label, hyperedges (list of lists of vertex ids)
    """
    records = []
    with open(path, "r") as f:
        lines = f.readlines()

    pos = 0
    n = int(lines[pos].strip())
    pos += 1

    for _ in range(n):
        header = lines[pos].strip().split()
        pos += 1
        num_v, num_e = int(header[0]), int(header[1])
        label_tokens = header[2:]
        if len(label_tokens) != 1:
            raise NotImplementedError(
                "Multi-label k-GWL variants are not supported yet "
                f"(got {len(label_tokens)} label tokens)."
            )
        label = int(label_tokens[0])

        # Vertex-label line: parsed to advance the reader only.
        pos += 1

        hyperedges = []
        for _ in range(num_e):
            vertex_ids = [int(v) for v in lines[pos].strip().split()]
            pos += 1
            hyperedges.append(vertex_ids)

        records.append(
            {"num_v": num_v, "num_e": num_e, "label": label, "hyperedges": hyperedges}
        )

    return records


def _compute_degree_features(num_v, hyperedges):
    degree = torch.zeros(num_v, dtype=torch.float)
    for edge in hyperedges:
        for v in edge:
            degree[v] += 1.0
    return degree.unsqueeze(1)


def _build_hypergraph_data(record, x_mode):
    num_v = record["num_v"]
    hyperedges = record["hyperedges"]

    if x_mode == "ones":
        x = torch.ones(num_v, 1)
    elif x_mode == "degree":
        x = _compute_degree_features(num_v, hyperedges)
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")

    node_ids, edge_ids = [], []
    for e_idx, edge in enumerate(hyperedges):
        for v in edge:
            node_ids.append(v)
            edge_ids.append(e_idx)

    hyperedge_index = torch.tensor([node_ids, edge_ids], dtype=torch.long)

    return Data(
        x=x,
        hyperedge_index=hyperedge_index,
        y=torch.tensor([record["label"]], dtype=torch.long),
        num_nodes=num_v,
    )


def _build_clique_data(record, x_mode):
    num_v = record["num_v"]
    hyperedges = record["hyperedges"]

    if x_mode == "ones":
        x = torch.ones(num_v, 1)
    elif x_mode == "degree":
        x = _compute_degree_features(num_v, hyperedges)
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")

    pair_set = set()
    for edge in hyperedges:
        verts = sorted(set(edge))
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                pair_set.add((verts[i], verts[j]))

    src, dst = [], []
    for u, v in pair_set:
        src.append(u)
        dst.append(v)
        src.append(v)
        dst.append(u)

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([record["label"]], dtype=torch.long),
        num_nodes=num_v,
    )


class KGWLGraphClassificationDataset(InMemoryDataset):
    """
    Graph-classification dataset built from a k-GWL hypergraph benchmark txt
    file (see NAME_TO_FILE for supported names).
    """

    def __init__(
        self,
        root,
        name,
        variant="hypergraph",
        x_mode="ones",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if name not in NAME_TO_FILE:
            raise ValueError(
                f"Unknown k-GWL dataset name: {name}. "
                f"Supported: {sorted(NAME_TO_FILE.keys())}"
            )
        if variant not in ("hypergraph", "clique"):
            raise ValueError(f"Unknown variant: {variant}")
        if x_mode not in ("ones", "degree"):
            raise ValueError(f"Unknown x_mode: {x_mode}")

        self.name = name
        self.variant = variant
        self.x_mode = x_mode

        super().__init__(root, transform, pre_transform, pre_filter)
        # torch>=2.6 defaults torch.load to weights_only=True, which blocks
        # PyG's DataEdgeAttr from unpickling. Safe here: this file is one we
        # generated ourselves in process().
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        # Raw k-GWL txt files live directly under `root` (grouped by dataset
        # subfolder), not under a `root/raw` convention, and are not
        # downloadable, so point straight at them.
        return self.root

    @property
    def raw_file_names(self):
        return [NAME_TO_FILE[self.name]]

    @property
    def processed_file_names(self):
        pre_transform_tag = repr(self.pre_transform).replace("\n", "") if self.pre_transform is not None else "none"
        return [f"data_{self.name}_{self.variant}_{self.x_mode}_{pre_transform_tag}.pt"]

    def download(self):
        raise FileNotFoundError(
            f"Expected raw k-GWL file at {self.raw_paths[0]}, but it is missing. "
            "This dataset is not downloadable; place the benchmark txt file there."
        )

    def process(self):
        records = _parse_kgwl_file(self.raw_paths[0])

        build_fn = _build_hypergraph_data if self.variant == "hypergraph" else _build_clique_data
        data_list = [build_fn(record, self.x_mode) for record in records]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def num_classes(self):
        return int(self.data.y.max().item()) + 1
