# Federated Learning · 4-Protocol Comparison Platform

Four communication protocols (**DCTCP / LTP / PLOT / PILT**) running
end-to-end BSP federated learning on the same NS3-AI wireless channel,
the same dataset and the same model initialisation. Ships with an
independent C++ protocol library (`cpp_proto/`) and a single-file,
interactive Plotly dashboard.

```
        ┌─────────────────────────┐
        │ NS3-AI driver  (Python) │ ── results/metrics_*.npz ─────┐
        └─────────────────────────┘                                │
        ┌─────────────────────────┐                                ▼
        │ C++ library (cpp_proto/)│ ── viz/proto_bench.json ──► viz/dashboard.html
        └─────────────────────────┘
```

---

## Quick Start

```bash
# C++ library + bench + single-file dashboard
cd cpp_proto
bash run_demo.sh
# → open http://localhost:8080/viz/dashboard.html in a browser
```

`run_demo.sh` runs `cmake build → ctest → ./compare_protocols → pack JSON
→ start an HTTP server`. If `results/metrics_*.npz` already exists from
the simulation driver it is folded into the dashboard automatically.

---

## Simulation driver (NS3-AI BSP)

```bash
./setup_ns3ai.sh                              # build the NS-3 module (once)

# Single protocol
python3 main.py --protocols pilt --workers 10 --model resnet50 \
                --time_limit 1200 --seed 0

# All four protocols (sequential)
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
| `--seed` | 0 | global RNG seed |

---

## Reference Numbers

ResNet-50, K=10, CIFAR-100, time_limit=1200 s, seed=0, identical NS3-AI
channel for all four protocols:

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
round-trip, K=10, 30 rounds):

| protocol | bytes_sent | delivery_correlation | encode_ms |
|---|---:|---:|---:|
| DCTCP | 28.3 MB | 1.000 |  7.4 |
| LTP   | 28.3 MB | 0.896 | 41.7 |
| PLOT  | 56.5 MB | 1.000 |  9.9 |
| PILT  | 0.81 MB | 0.859 |  9.0 |

Live data is read by `cpp_proto/viz/dashboard.html` (auto-loads the most
recent `.npz` outputs and the bench JSON).

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
│   ├── pilt_protocol.py      #   importance EMA + ranked ε_l + top-|g| + EF + GA
│   ├── ltp_protocol.py       #   Early-Close + Random-K bubble-fill
│   └── plot_protocol.py      #   per-layer LTT + retx
│   # DCTCP is inlined in main.py (reliable full-grad)
│
├── federated/                # PyTorch FL building blocks
│   ├── dataset.py            #   CIFAR-100 + Dirichlet non-IID split
│   └── model_torch.py        #   ResNet/VGG + FLWorker + ParameterServer
│
├── cpp_proto/                # standalone C++ protocol library + bench + dashboard
│   ├── include/              #   pilt/ dctcp/ ltp/ plot/ common/
│   ├── src/                  #   PILT implementation (others are header-only)
│   ├── tests/                #   ctest, 4 test binaries
│   ├── demo/                 #   PILT TCP end-to-end demo
│   ├── bench/                #   compare_protocols (4-protocol head-to-head)
│   ├── viz/                  #   dashboard.html + sim_to_json.py
│   ├── CMakeLists.txt
│   └── run_demo.sh           #   one-shot build+test+bench+serve
│
└── venv/                     # optional Python virtual environment
```

`results/`, `data/`, `logs/` and `cpp_proto/build/` are created on
demand by the runners and are intentionally not part of the source tree.

---

## Protocol entry points

### Python (NS3-AI driver)

```python
from protocols.pilt_protocol import PILTProtocol, PILTConfig, GAConfig
from protocols.ltp_protocol  import LTPProtocol
from protocols.plot_protocol import PLOTProtocol
# main.py wires the four protocols into a single BSP loop;
# select with --protocols.
```

### C++ (embeddable)

```cpp
#include "pilt/encoder.hpp"      // pilt::PILTEncoder / PILTAggregator
#include "dctcp/protocol.hpp"    // dctcp::Encoder / Aggregator
#include "ltp/protocol.hpp"      // ltp::Encoder / Aggregator + decide_close()
#include "plot/protocol.hpp"     // plot::Encoder / Aggregator (two-pass)
#include "common/tcp.hpp"        // POSIX TCP helpers + try_set_dctcp(fd)
```

See `cpp_proto/README.md` for the full API.

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
