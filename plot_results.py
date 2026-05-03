#!/usr/bin/env python3
"""plot_results.py — render comparison figures from main.py outputs.

Generates:
  acc_vs_time      results/pilt_fig2_acc_vs_time.png
  tail_fct         results/pilt_fig4_tail_fct.png
  diagnostics      results/pilt_extra_diagnostics.png

Usage:
    python3 plot_results.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLORS = {
    "dctcp": "#D62728",   # red
    "ltp"  : "#1F77B4",   # blue
    "plot" : "#2CA02C",   # green
    "pilt" : "#FF7F0E",   # orange
}
LABELS = {"dctcp": "DCTCP", "ltp": "LTP", "plot": "PLOT", "pilt": "PILT"}
LINESTYLES = {"dctcp": "--", "ltp": "-.", "plot": ":", "pilt": "-"}
PROTO_ORDER = ["dctcp", "ltp", "plot", "pilt"]


def load(proto: str):
    p = os.path.join(RESULTS_DIR, f"metrics_{proto}.npz")
    if not os.path.exists(p):
        return None
    return dict(np.load(p, allow_pickle=True))


def fig2_acc_vs_time(data: dict, outpath: str):
    """Test accuracy vs. wall-clock time."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for proto in PROTO_ORDER:
        m = data.get(proto)
        if m is None:
            continue
        t = m["wall_ms"] / 1000.0
        lw = 2.5 if proto == "pilt" else 2.0
        ax.plot(t, m["test_acc"], color=COLORS[proto],
                ls=LINESTYLES[proto], lw=lw,
                label=LABELS[proto], marker="o", ms=3, alpha=0.9)
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs. training time (K=10)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"  saved {outpath}")


def fig4_tail_fct(data: dict, outpath: str):
    """Normalised 99th percentile tail FCT."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    base = data.get("dctcp")
    if base is None:
        print("  (no DCTCP baseline; skipping normalisation)")
        base_p99 = 1.0
    else:
        base_p99 = float(np.quantile(base["per_worker_fct_ms"].ravel(), 0.99))

    names, p99s, p95s, colors = [], [], [], []
    for proto in PROTO_ORDER:
        m = data.get(proto)
        if m is None:
            continue
        fct = m["per_worker_fct_ms"].ravel()
        p99 = float(np.quantile(fct, 0.99))
        p95 = float(np.quantile(fct, 0.95))
        names.append(LABELS[proto])
        p99s.append(p99 / base_p99)
        p95s.append(p95 / base_p99)
        colors.append(COLORS[proto])

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, p99s, w, color=colors, label="99th pct.", alpha=0.85)
    ax.bar(x + w/2, p95s, w, color=colors, label="95th pct.",
           alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Normalised tail FCT  (vs. DCTCP)")
    ax.set_title("Tail FCT across protocols (K=10)")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    for i, (p99, name) in enumerate(zip(p99s, names)):
        ax.text(i - w/2, p99 + 0.02, f"{p99:.2f}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"  saved {outpath}")


def extra_diagnostics(data: dict, outpath: str):
    """Two-panel: (a) per-round time, (b) PILT makespan evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for proto in PROTO_ORDER:
        m = data.get(proto)
        if m is None:
            continue
        rounds = m["rounds"]
        ax.plot(rounds, m["round_ms"], color=COLORS[proto],
                ls=LINESTYLES[proto], lw=1.5, alpha=0.85,
                label=LABELS[proto])
    ax.set_xlabel("BSP round")
    ax.set_ylabel("Round duration (ms)")
    ax.set_title("(a) Per-round completion time")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    m = data.get("pilt")
    if m is not None and "makespan_slots" in m:
        ms = m["makespan_slots"]
        ax.plot(np.arange(1, len(ms) + 1), ms, color=COLORS["pilt"],
                ls=LINESTYLES["pilt"], lw=1.8,
                label=f"PILT (mean={np.mean(ms):.1f})")
        ax.set_xlabel("BSP round")
        ax.set_ylabel("Wireless makespan (slots = ms)")
        ax.set_title("(b) GA-solved makespan per round")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"  saved {outpath}")


def _time_to_accuracy(m, target: float) -> float | None:
    """Return the earliest sim-seconds at which test_acc ≥ target, else None."""
    acc = np.asarray(m["test_acc"])
    t   = np.asarray(m["wall_ms"]) / 1000.0
    idx = np.where(acc >= target)[0]
    return float(t[idx[0]]) if idx.size > 0 else None


def summary_table(data: dict, targets=(0.10, 0.20, 0.30, 0.40, 0.50)):
    """
    Runs are TIME-bounded (same wall-clock budget), so final JCT ≈ time_limit
    for every protocol and a normalised-JCT column would be meaningless.

    We report:
      * Rounds completed within the budget
      * Final accuracy at the budget
      * Time-to-accuracy for representative targets
      * p99 flow completion time (normalised to DCTCP)
    """
    base = data.get("dctcp")
    base_p99 = float(np.quantile(base["per_worker_fct_ms"].ravel(), 0.99)) \
               if base is not None else None

    tgt_header = " ".join([f"t≥{int(t*100)}%(s)".rjust(10) for t in targets])
    header = (f"  {'Protocol':<8}{'Rounds':>8}{'Wall(s)':>10}"
              f"{'FinalAcc':>10} {tgt_header} "
              f"{'p99 FCT ms':>12} {'Norm p99':>10}")
    print("\n" + "═" * len(header))
    print(header)
    print("═" * len(header))
    for proto in PROTO_ORDER:
        m = data.get(proto)
        if m is None:
            continue
        wall = float(m["wall_ms"][-1] / 1000.0)
        acc  = float(m["test_acc"][-1])
        rds  = int(len(m["rounds"]))
        p99  = float(np.quantile(m["per_worker_fct_ms"].ravel(), 0.99))
        np99 = p99 / base_p99 if base_p99 else float("nan")
        tta  = []
        for t in targets:
            s = _time_to_accuracy(m, t)
            tta.append(f"{s:>10.1f}" if s is not None else f"{'—':>10}")
        print(f"  {LABELS[proto]:<8}{rds:>8}{wall:>10.1f}"
              f"{acc:>10.3f} {' '.join(tta)} {p99:>12.1f}{np99:>10.3f}")
    print("═" * len(header))

    # Speed-up vs LTP at acc=30 %.
    ref_proto, ref_t = "ltp", None
    m_ref = data.get(ref_proto)
    if m_ref is not None:
        ref_t = _time_to_accuracy(m_ref, 0.30)
    if ref_t is not None:
        print("\nSpeed-up at acc=30 %   (t_LTP / t_this):")
        for proto in PROTO_ORDER:
            m = data.get(proto)
            if m is None:
                continue
            s = _time_to_accuracy(m, 0.30)
            if s is None or s <= 0:
                print(f"    {LABELS[proto]:<8}  did not reach 30 %")
            else:
                print(f"    {LABELS[proto]:<8}  {ref_t/s:>5.2f} × vs LTP")


def main():
    data = {}
    for proto in PROTO_ORDER:
        d = load(proto)
        if d is not None:
            data[proto] = d
    if not data:
        print("No results found. Run  python main.py  first.")
        return
    print(f"Loaded protocols: {list(data.keys())}")
    fig2_acc_vs_time(data,  os.path.join(RESULTS_DIR, "pilt_fig2_acc_vs_time.png"))
    fig4_tail_fct  (data,   os.path.join(RESULTS_DIR, "pilt_fig4_tail_fct.png"))
    extra_diagnostics(data, os.path.join(RESULTS_DIR, "pilt_extra_diagnostics.png"))
    summary_table(data)


if __name__ == "__main__":
    main()
