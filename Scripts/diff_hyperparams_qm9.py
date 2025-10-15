import os
import re
import wandb
from collections import defaultdict
from Misc.config import config

OUTPUT_DIR = "Configs/QM9_Run2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex to capture qm9_* dataset case-insensitively
QM9_NAME_RE = re.compile(r"^umi_(qm9_\d+)", re.IGNORECASE)

def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def best_runs_by_dataset(api):
    sweeps = api.project(config.project).sweeps()
    best = defaultdict(lambda: {"mae": float("inf"), "run": None})

    for sweep in sweeps:
        name = getattr(sweep, "name", None)
        if not (name and name.lower().startswith("umi_qm9")):
            continue

        m = QM9_NAME_RE.match(name)
        if not m:
            continue
        dataset = m.group(1)  # e.g., "QM9_0", "qm9_11"

        for run in sweep.runs:
            mae = run.summary.get("Final/Val/mae")
            try:
                mae_val = float(mae)
            except (TypeError, ValueError):
                continue

            if mae_val < best[dataset]["mae"]:
                best[dataset] = {"mae": mae_val, "run": run}

    return best

def format_literal(payload):
    lines = ["{"]
    for i, (k, v) in enumerate(payload.items()):
        lines.append(f"  {k}: {v}, ")
    lines[-1] = lines[-1].rstrip(", ")
    lines.append("}")
    return "\n".join(lines)

def build_payload(cfg):
    payload = {}

    # Scalars
    for key in ["batch_size", "model", "scheduler", "epochs", "ff", "residual", "tracking"]:
        if key == "batch_size":
            payload[key] = "{'values': [64, 1024]}"
        elif key in cfg:
            payload[key] = f"{{'value': {repr(cfg[key])}}}"
        else:
            payload[key] = "{'value': None}"

    # Lists
    for key in ["emb_dim", "drop_out", "num_mp_layers"]:
        if key in cfg:
            vals = cfg[key] if isinstance(cfg[key], list) else [cfg[key]]
            payload[key] = f"{{'values': {repr(vals)}}}"
        else:
            payload[key] = "{'values': []}"

    # Pooling: best + nodesum
    best_pool = cfg.get("pooling")
    pools = dedup_preserve_order([best_pool, "nodesum"])
    payload["pooling"] = f"{{'values': {repr(pools)}}}"

    return payload

def main():
    api = wandb.Api()
    best = best_runs_by_dataset(api)

    for dataset, rec in best.items():
        run = rec["run"]
        if run is None:
            continue

        cfg = dict(run.config or {})
        payload = build_payload(cfg)

        # Extract numeric variant safely, case-insensitive
        variant_match = re.match(r"qm9_(\d+)", dataset, re.IGNORECASE)
        if not variant_match:
            continue
        variant = variant_match.group(1)

        out_path = os.path.join(OUTPUT_DIR, f"QM9_{variant}.yaml")

        with open(out_path, "w") as f:
            f.write(format_literal(payload))

        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
