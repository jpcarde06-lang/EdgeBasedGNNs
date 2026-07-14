"""
Measures cold-cache preprocessing time for KGWLGraphClassificationDataset
with variant='hypergraph', x_mode='ones', pre_transform=EBHNNTransform().

For each dataset, the existing processed cache is deleted first so the
timing reflects a fresh Data-list build + EBHNN transform + collate, not a
cached torch.load.

Run from EdgeBasedGNNs/ with:
    PYTHONPATH=. python Exp/measure_preprocessing.py
"""

import glob
import json
import os
import time

from Misc.config import config
from Misc.EBHNN_trafo import EBHNNTransform
from Misc.kgwl_datasets import KGWLGraphClassificationDataset

DATASETS = ["steam_player", "twitter_friend", "IMDB_dir_form"]
VARIANT = "hypergraph"
X_MODE = "ones"
RESULTS_PATH = "Results/preprocessing_times.json"


class _CompatEBHNNTransform(EBHNNTransform):
    """
    PyG-version compat shim: newer torch_geometric makes BaseTransform an ABC
    requiring `forward`, which EBHNNTransform (written for older PyG) doesn't
    define. EBHNNTransform overrides `__call__` directly, so it takes
    precedence over BaseTransform's in the MRO and this stub is never called.
    See Tests/test_ebhnn_trafo.py for the original pattern.
    """

    def forward(self, data):  # pragma: no cover - dead code, see class docstring
        raise NotImplementedError


def _delete_processed_cache(name):
    processed_dir = os.path.join(config.KGWL_DATA_PATH, "processed")
    pattern = os.path.join(processed_dir, f"data_{name}_{VARIANT}_{X_MODE}_*.pt")
    removed = []
    for path in glob.glob(pattern):
        os.remove(path)
        removed.append(path)
    return removed


def _measure(name):
    removed = _delete_processed_cache(name)
    print(f"[{name}] removed cached files: {removed or 'none'}")

    start = time.perf_counter()
    dataset = KGWLGraphClassificationDataset(
        root=config.KGWL_DATA_PATH,
        name=name,
        variant=VARIANT,
        x_mode=X_MODE,
        pre_transform=_CompatEBHNNTransform(),
    )
    elapsed = time.perf_counter() - start

    num_graphs = len(dataset)
    total_active_pairs = 0
    total_wl_rows = 0
    total_wt_rows = 0
    total_wr_rows = 0
    for data in dataset:
        total_active_pairs += data.edge_index.size(1)
        total_wl_rows += data.wl_mapping.size(0)
        total_wt_rows += data.wt_mapping.size(0)
        total_wr_rows += data.wr_mapping.size(0)

    return {
        "construction_time_sec": elapsed,
        "num_graphs": num_graphs,
        "total_active_pairs": total_active_pairs,
        "total_wl_rows": total_wl_rows,
        "total_wt_rows": total_wt_rows,
        "total_wr_rows": total_wr_rows,
    }


def main():
    results = {}
    for name in DATASETS:
        results[name] = _measure(name)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    header = f"{'dataset':<16}{'time_sec':>10}{'graphs':>9}{'active_pairs':>14}{'wl_rows':>12}{'wt_rows':>12}{'wr_rows':>12}"
    print()
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<16}{r['construction_time_sec']:>10.3f}{r['num_graphs']:>9}"
            f"{r['total_active_pairs']:>14}{r['total_wl_rows']:>12}"
            f"{r['total_wt_rows']:>12}{r['total_wr_rows']:>12}"
        )
    print()
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
