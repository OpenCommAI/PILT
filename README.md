# PILT — Probabilistic Importance-Layered Transport for Federated Learning

**PILT** is a transport / scheduling co-design for synchronous (BSP)
federated learning over a shared wireless channel. It treats every BSP
round as a constrained allocation problem on a 2-D time-frequency
resource-block (RB) grid and asks:

> *Given a tight per-round byte budget, which layers from which workers
> should we send, in what order, on which RBs, so that the recovered
> aggregate is as close to the true average as possible?*

The repository ships the full PILT design (algorithm + transport
encoder + GA scheduler + C++ library) together with three reference
baselines (DCTCP / LTP / PLOT) that share the same NS3-AI channel,
dataset and model — so PILT's design choices can be measured rather
than asserted.

```
        ┌─────────────────────────┐
        │ NS3-AI driver  (Python) │ ── results/metrics_*.npz ─────┐
        └─────────────────────────┘                                │
        ┌─────────────────────────┐                                ▼
        │ C++ library (cpp_proto/)│ ── viz/proto_bench.json ──► viz/dashboard.html
        └─────────────────────────┘
```

---

## PILT design at a glance

PILT has four cooperating components. Each one is replaceable; together
they define what we call PILT.

| component | symbol | role |
|---|---|---|
| **Importance tracker** | `v_l ← β·v_l + (1-β)·‖g_l‖₂/√S_l` | per-layer L2-norm EMA, RMS-normalised across layer sizes |
| **Ranked ε-allocator** | `ε_l = f(rank_l, E_total, ε_min)` | distributes the round byte budget `E_total` across layers by importance rank, with a floor `ε_min` so unimportant layers still drift |
| **Top-\|g\| + EF encoder** | `pilt::PILTEncoder` | per-(worker, layer) selects the highest-magnitude entries that fit `ε_l·S_l` bytes, packs `(idx, val, CRC32)` sparse frames, and feeds the discarded mass back as an error-feedback residual the next round |
| **GA scheduler over RB grid** | `PILTScheduler` (dual-population GA) | jointly chooses the start-slot of every (worker, layer) frame on an `M`-wide RB grid, minimising makespan subject to per-RB exclusivity and PS overhead pipelining |

All four pieces live behind a single API:

```python
from protocols.pilt_protocol import PILTProtocol, PILTConfig, GAConfig
proto = PILTProtocol(n_workers=K, n_layers=L, layer_sizes=S,
                     cfg=PILTConfig(beta=0.9, d=0.05, E_total=0.5,
                                    eps_min=0.05, ga=GAConfig(...)))
eps         = proto.compute_ratios()              # ranked ε_l
budget      = proto.target_payload_bytes(eps)     # per-(k,l) bytes
start, T    = proto.solve_schedule(D)             # GA → start-slot table
plan        = proto.schedule_to_worker_plan(start, K)
enc, masks, sent = proto.worker_encode(k, grads_k, eps)
proto.update_importance(avg_layers)               # v_l for next round
```

The C++ side mirrors this:

```cpp
#include "pilt/encoder.hpp"
#include "pilt/aggregator.hpp"
pilt::EncoderConfig cfg{.E_total=0.5f, .eps_min=0.05f, .beta=0.9f, .d=0.05f};
pilt::PILTEncoder    enc({S0, S1, S2}, cfg);
pilt::PILTAggregator agg({S0, S1, S2});
```

See `protocols/pilt_protocol.py` and `cpp_proto/include/pilt/` for the
full source, and `cpp_proto/README.md` for the C++ API surface.

---

## Quick start

### C++ library + bench + interactive dashboard

```bash
cd cpp_proto
bash run_demo.sh
# → open http://localhost:8080/viz/dashboard.html in a browser
```

`run_demo.sh` runs `cmake build → ctest → ./compare_protocols → pack
JSON → start an HTTP server`. If `results/metrics_*.npz` already exists
from the simulation driver it is folded into the dashboard
automatically.

### NS3-AI BSP simulation driver

```bash
./setup_ns3ai.sh                              # build the NS-3 module (once)

# Run PILT on its own
python3 main.py --protocols pilt --workers 10 --model resnet50 \
                --time_limit 1200 --seed 0

# Run PILT alongside the three baselines (sequential)
python3 main.py --protocols dctcp,ltp,plot,pilt --workers 10 \
                --model resnet50 --time_limit 1200 --seed 0
```

Outputs are saved to `results/metrics_<protocol>.npz`:

| key | meaning |
|---|---|
| `wall_ms` | cumulative simulated wall-clock (ms), GA scheduling included |
| `round_ms` | per-BSP-round duration (ms) |
| `rounds` | BSP round numbers |
| `test_acc` / `test_loss` | global test accuracy / loss (every few rounds) |
| `ps_overhead_ms` | intrinsic PS-side scheduling time |
| `ps_exposed_ms` | PS overhead remaining after pipelining with worker SGD |

### Common flags

| flag | default | meaning |
|---|---|---|
| `--protocols` | `dctcp,ltp,plot,pilt` | comma-separated subset to run |
| `--workers` | 10 | K |
| `--model` | `resnet20` | `resnet20 / resnet50 / vgg16` |
| `--time_limit` | 180 | simulated wall-clock budget (seconds) |
| `--rounds` | 100000 | hard cap on rounds (whichever hits first) |
| `--pilt_n_rbs` | 10 | M, time-frequency RB grid size |
| `--pilt_pipeline_ga` | 1 | overlap GA with worker SGD |
| `--autotune` | off | enable the built-in adjudicator (see below) |
| `--autotune_target` | 0.9 | target ratio `T_GA / T_cmp ∈ (0,1]` for the adjudicator |
| `--autotune_max_E` | 16 | cap on `local_steps` the adjudicator may pick |
| `--seed` | 0 | global RNG seed |

### Built-in adjudicator (内置判决模块, opt-in)

When `--autotune` is set AND the protocol is `pilt`, the driver runs a
short startup micro-benchmark before the BSP loop:

1. Times one real `worker.compute_gradient(...)` pass on the slowest
   (largest-shard) worker — this is `T_cmp`.
2. Times the dual-population GA solver on the live scheduling instance
   for a small grid of `(Np, G_max)` candidates — this is `T_GA(Np,G_max)`.
3. Picks the largest `(Np, G_max)` such that
   `T_GA · safety_margin ≤ target_overlap · T_cmp`. If no candidate fits
   even at the smallest grid point, the calibrator bumps `local_steps`
   (capped at `--autotune_max_E`) to extend the compute mask, and
   rescales the simulator's `worker_compute_ms` accordingly so the BSP
   accounting stays consistent.

Effect: when GA / SGD pipelining is on (default), `ps_exposed_ms → 0`
and the GA's intrinsic wall-time is hidden inside worker compute — the
algorithm overhead never surfaces on the round critical path.  Off by
default — users opt in.  Implementation: `protocols/autotuner.py`
(`OverlapAutoTuner`, `AutoTuneConfig`, `AutoTuneReport`).

---

## Validating the design

ResNet-50, K=10, CIFAR-100, time_limit=1200 s, seed=0, identical NS3-AI
channel for all four protocols. The three baselines (DCTCP / LTP /
PLOT) ablate the PILT design choices: DCTCP keeps the full gradient,
LTP drops bytes randomly under a deadline, PLOT does per-layer LTT but
without importance ranking or RB scheduling.

| protocol | rounds | final acc | final loss | mean round_ms | p99 FCT (ms) |
|---|---:|---:|---:|---:|---:|
| DCTCP | 52  | 0.011 | 4.839 | 23187 | 23245 |
| LTP   | 300 | 0.139 | 3.660 |  4008 |  3827 |
| PLOT  | 231 | 0.135 | 3.673 |  5196 |  5370 |
| **PILT** | **303** | **0.176** | **3.434** | **3965** | **1957** |

Time-to-target (simulated seconds needed to reach a given accuracy):

| target acc | DCTCP | LTP | PLOT | PILT | PILT vs LTP |
|---:|---:|---:|---:|---:|---:|
| 0.05  | — | 200.3 | 230.3 | **173.8** | **1.15×** |
| 0.10  | — | 501.0 | 543.4 | **407.9** | **1.23×** |
| 0.13  | — | 821.4 | 908.5 | **585.0** | **1.40×** |
| 0.135 | — | 901.5 | 1168.0 | **644.1** | **1.40×** |
| 0.14  | — |  n/a |  n/a | 664.2 | — |
| 0.17  | — |  n/a |  n/a | 823.9 | — |

C++ library head-to-head bench (synthetic gradients, single-machine
round-trip, K=10, 30 rounds) — confirms PILT's wire footprint advantage
in isolation:

| protocol | bytes_sent | delivery_correlation | encode_ms |
|---|---:|---:|---:|
| DCTCP | 28.3 MB | 1.000 |  7.4 |
| LTP   | 28.3 MB | 0.896 | 41.7 |
| PLOT  | 56.5 MB | 1.000 |  9.9 |
| PILT  | 0.81 MB | 0.859 |  9.0 |

Live data is read by `cpp_proto/viz/dashboard.html` (auto-loads the
most recent `.npz` outputs and the bench JSON).

---

## Layout

```
FL/
├── main.py                   # NS3-AI BSP simulation entry point
├── plot_results.py           # offline matplotlib comparison figures
├── config.py                 # simulation hyper-parameters
├── setup_ns3ai.sh            # build the NS-3 module
├── requirements.txt
│
├── protocols/                # algorithm-side implementations
│   ├── pilt_protocol.py      #   PILT: importance EMA + ranked ε_l + top-|g| + EF + GA
│   ├── ltp_protocol.py       #   baseline: Early-Close + Random-K bubble-fill
│   ├── plot_protocol.py      #   baseline: per-layer LTT + retx
│   └── autotuner.py          #   opt-in adjudicator: probes T_cmp / T_GA at startup,
│                             #   tunes local_steps + GA Np/G_max for overlap hiding
│   # DCTCP baseline is inlined in main.py (reliable full-grad)
│
├── federated/                # PyTorch FL building blocks
│   ├── dataset.py            #   CIFAR-100 + Dirichlet non-IID split
│   └── model_torch.py        #   ResNet/VGG + FLWorker + ParameterServer
│
├── cpp_proto/                # standalone C++ protocol library + bench + dashboard
│   ├── include/pilt/         #   PILT C++ headers (encoder / aggregator / wire)
│   ├── include/{dctcp,ltp,plot}/  baselines, header-only
│   ├── src/                  #   PILT implementation
│   ├── tests/                #   ctest, 4 test binaries
│   ├── demo/                 #   PILT TCP end-to-end demo
│   ├── bench/                #   compare_protocols (PILT vs 3 baselines)
│   ├── viz/                  #   dashboard.html + sim_to_json.py
│   ├── CMakeLists.txt
│   └── run_demo.sh           #   one-shot build+test+bench+serve
│
└── venv/                     # optional Python virtual environment
```

`results/`, `data/`, `logs/` and `cpp_proto/build/` are created on
demand by the runners and are intentionally not part of the source
tree.

---

## Dependencies

```bash
# Python (simulation side)
pip install -r requirements.txt
# core: torch, torchvision, numpy, matplotlib

# C++ (cpp_proto/)
# requires: CMake ≥ 3.16, GCC/Clang with C++17, pthread
```

NS-3 / NS3-AI installation is handled by `setup_ns3ai.sh`.
