#!/usr/bin/env python3
"""sim_to_json.py — pack sim outputs + C++ bench into one JSON.

Combines `results/metrics_*.npz` (NS3-AI sim per protocol) and a
`compare_protocols` JSON file into the single self-describing file that
`dashboard.html` consumes.

Usage:

    python3 viz/sim_to_json.py \\
        --sim_dir   ../results       \\
        --bench     /tmp/proto_bench.json \\
        --out       viz/dashboard_data.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np


PROTO_PRETTY = {"dctcp": "DCTCP", "ltp": "LTP", "plot": "PLOT", "pilt": "PILT"}
PROTO_COLOR  = {"DCTCP": "#d62728", "LTP": "#ff7f0e",
                "PLOT": "#9467bd", "PILT": "#1f77b4"}


def _load_sim_npz(sim_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for short in ("dctcp", "ltp", "plot", "pilt"):
        path = os.path.join(sim_dir, f"metrics_{short}.npz")
        if not os.path.isfile(path):
            print(f"[sim_to_json] skip missing {path}", file=sys.stderr)
            continue
        d = np.load(path, allow_pickle=True)
        sim_s   = (d["wall_ms"] / 1000.0).astype(float).tolist()
        rounds  = d["rounds"].astype(int).tolist()
        acc     = d["test_acc"].astype(float).tolist()
        loss    = d["test_loss"].astype(float).tolist()
        rt_ms   = d["round_ms"].astype(float).tolist()
        ps_ovh  = d["ps_overhead_ms"].astype(float).tolist() if "ps_overhead_ms" in d.files else []
        ps_exp  = d["ps_exposed_ms"].astype(float).tolist()  if "ps_exposed_ms" in d.files else []
        out.append({
            "name":        PROTO_PRETTY[short],
            "color":       PROTO_COLOR[PROTO_PRETTY[short]],
            "sim_seconds": sim_s,
            "rounds":      rounds,
            "test_acc":    acc,
            "test_loss":   loss,
            "round_ms":    rt_ms,
            "ps_overhead_ms": ps_ovh,
            "ps_exposed_ms":  ps_exp,
            "final_acc":   float(acc[-1]) if acc else 0.0,
            "final_loss":  float(loss[-1]) if loss else 0.0,
            "total_rounds": len(rounds),
            "max_sim_seconds": float(sim_s[-1]) if sim_s else 0.0,
            "mean_round_ms":  float(np.mean(rt_ms)) if rt_ms else 0.0,
        })
    return out


def _common_checkpoints(sim: List[Dict[str, Any]]) -> List[float]:
    if not sim:
        return []
    max_t = max(p["max_sim_seconds"] for p in sim)
    if max_t <= 0:
        return []
    # 12 evenly-spaced grid points across [0, max_t]
    return [round(max_t * (i + 1) / 12.0, 1) for i in range(12)]


def _interp_table(sim: List[Dict[str, Any]],
                  checkpoints: List[float]) -> Dict[str, List[Any]]:
    """Build a 'same sim-time' comparison table."""
    table: Dict[str, List[Any]] = {"sim_seconds": checkpoints}
    for p in sim:
        col_acc, col_loss = [], []
        sim_x = np.asarray(p["sim_seconds"]); acc = np.asarray(p["test_acc"])
        los   = np.asarray(p["test_loss"])
        for cp in checkpoints:
            if not sim_x.size or cp > sim_x[-1] + 1e-3:
                col_acc.append(None); col_loss.append(None)
            else:
                col_acc.append(float(np.interp(cp, sim_x, acc)))
                col_loss.append(float(np.interp(cp, sim_x, los)))
        table[p["name"] + "_acc"]  = col_acc
        table[p["name"] + "_loss"] = col_loss
    return table


def _time_to_target(sim: List[Dict[str, Any]],
                    targets: List[float]) -> Dict[str, List[Any]]:
    """For each protocol and accuracy target, the smallest sim-time at
    which test_acc first reached the target.  None if never reached."""
    out: Dict[str, List[Any]] = {"target_acc": targets}
    for p in sim:
        col: List[Any] = []
        sim_x = np.asarray(p["sim_seconds"])
        acc   = np.asarray(p["test_acc"])
        for tgt in targets:
            mask = acc >= tgt
            if not mask.any():
                col.append(None)
            else:
                col.append(float(sim_x[np.argmax(mask)]))
        out[p["name"]] = col
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default="results")
    ap.add_argument("--bench",   default=None,
                    help="Optional path to compare_protocols JSON.")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--targets", default="0.05,0.10,0.15,0.20,0.25",
                    help="Time-to-accuracy targets (comma-separated).")
    args = ap.parse_args()

    sim = _load_sim_npz(args.sim_dir)
    bench = None
    if args.bench and os.path.isfile(args.bench):
        with open(args.bench) as f:
            bench = json.load(f)

    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    out = {
        "schema":    "pilt-dashboard-v1",
        "sim":       sim,
        "compare_table":  _interp_table(sim, _common_checkpoints(sim)),
        "time_to_target": _time_to_target(sim, targets),
        "bench":     bench,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
