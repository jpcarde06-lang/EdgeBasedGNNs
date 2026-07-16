"""
k-GWL benchmark training script 
"""

import argparse
import json
import os
import time
import gc
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from Misc.config import config
from Misc.EBHNN_trafo import EBHNNTransform
from Misc.kgwl_datasets import KGWLGraphClassificationDataset
from Misc.utils import PredictionType
from Models.EBGNN import EBGNN
from Models.encoder import StampEncoder


class _CompatEBHNNTransform(EBHNNTransform):

    def forward(self, data):  # pragma: no cover - dead code, see class docstring
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(description="Train EB-HNN on a k-GWL benchmark dataset")
    parser.add_argument("--dataset", type=str, required=True,
                         help="steam_player | twitter_friend | IMDB_dir_form | IMDB_wri_form | "
                              "IMDB_dir_genre | IMDB_wri_genre")
    parser.add_argument("--model", type=str, default="EBHNN", help="Model to train (currently only EBHNN)")
    parser.add_argument("--x_mode", type=str, default="degree", choices=["ones", "degree"],
                         help="Vertex feature mode (protocol default: degree_as_tag=True)")
    parser.add_argument("--bins", type=int, default=8, help="Number of hyperedge-size stamp bins")
    parser.add_argument("--max_hyperedge_size", type=int, default=None,
                         help="Optional cap on hyperedge size; hyperedges with more members "
                              "than this are skipped entirely (OFF by default, no cap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--debug_one_fold", action="store_true",
                         help="Only run fold 0, then stop (for fast debugging/iteration)")
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # auto never picks mps: measured 25-55x slower than cpu on this workload
    # (many small variable-size scatter ops over active pairs/wt_mapping),
    # and appeared to degrade further epoch over epoch. Still selectable
    # explicitly via --device mps.
    return torch.device("cpu")


def build_model(args, num_classes):
    return EBGNN(
        num_classes=num_classes,
        num_tasks=1,
        num_layer=args.num_layers,
        emb_dim=args.emb_dim,
        gnn_type="ebhnn",
        residual=0,
        ff=0,
        drop_ratio=0.0,
        JK="last",
        graph_pooling="sum",
        node_encoder=torch.nn.Linear(1, args.emb_dim),
        edge_encoder=StampEncoder(args.emb_dim, num_size_bins=args.bins),
        num_mlp_layers=2,
        activation="relu",
        prediction_type=PredictionType.GRAPH_PREDICTION,
    )


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
        all_y.append(batch.y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    f1_micro = f1_score(y, preds, average="micro")
    f1_macro = f1_score(y, preds, average="macro")
    return acc, f1_micro, f1_macro


def overfit_one_batch_gate(args, dataset, device, num_classes):
    
    batch = next(iter(DataLoader(dataset[:4], batch_size=4, shuffle=False))).to(device)

    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    train_acc = 0.0
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, batch.y)

    if train_acc != 1.0:
        raise RuntimeError(
            f"Overfit-one-batch gate FAILED: train accuracy after 200 steps = {train_acc:.4f} "
            "(expected 1.0). Aborting before the fold loop -- fix model/data wiring before "
            "trusting any fold results."
        )
    print(f"Overfit-one-batch gate PASSED: train accuracy = {train_acc:.4f} after 200 steps")


def run_fold(args, dataset, train_idx, test_idx, device, num_classes, fold_idx):
    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    patience = 50
    max_epochs = 2000  # safety cap; patience-based early stopping should trigger first

    best_test_acc, best_epoch, best_f1_micro, best_f1_macro = -1.0, -1, 0.0, 0.0
    epochs_without_improvement = 0
    epoch = 0

    while epochs_without_improvement < patience and epoch < max_epochs:
        epoch_t0 = time.perf_counter()
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        test_acc, f1_micro, f1_macro = evaluate(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch
            best_f1_micro, best_f1_macro = f1_micro, f1_macro
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_dt = time.perf_counter() - epoch_t0
        print(f"  [fold {fold_idx}] epoch {epoch}: test_acc={test_acc:.4f} "
              f"(best={best_test_acc:.4f}@{best_epoch}) time={epoch_dt:.2f}s "
              f"no_improve={epochs_without_improvement}/{patience}", flush=True)
        epoch += 1

    print(f"[fold {fold_idx}] best_test_acc={best_test_acc:.4f} at epoch {best_epoch} (ran {epoch} epochs)")
    results = {
        "fold": fold_idx,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "f1_micro": best_f1_micro,
        "f1_macro": best_f1_macro,
        "epochs_run": epoch,
    }
    del model, optimizer, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    return results

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model.upper() != "EBHNN":
        raise NotImplementedError(f"Model '{args.model}' is not wired into this training script yet.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = KGWLGraphClassificationDataset(
        root=config.KGWL_DATA_PATH,
        name=args.dataset,
        variant="hypergraph",
        x_mode=args.x_mode,
        pre_transform=_CompatEBHNNTransform(num_size_bins=args.bins,
                                             max_hyperedge_size=args.max_hyperedge_size),
    )
    num_classes = dataset.num_classes
    print(f"Loaded {args.dataset}: {len(dataset)} graphs, num_classes={num_classes}")

    overfit_one_batch_gate(args, dataset, device, num_classes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config.RESULTS_PATH, args.dataset, args.model)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{timestamp}.json")

    results = {"dataset": args.dataset, "model": args.model, "args": vars(args), "folds": []}

    def save():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        fold_result = run_fold(args, dataset, train_idx, test_idx, device, num_classes, fold_idx)
        results["folds"].append(fold_result)
        save()  # persist as each fold completes, not only at the end

        if args.debug_one_fold:
            break

    accs = [f["best_test_acc"] for f in results["folds"]]
    if len(accs) > 1:
        results["mean_test_acc"] = float(np.mean(accs))
        results["std_test_acc"] = float(np.std(accs))
        save()
        print(f"mean best-test-acc over {len(accs)} folds: "
              f"{results['mean_test_acc']:.4f} +/- {results['std_test_acc']:.4f}")

    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
