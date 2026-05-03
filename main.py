#!/usr/bin/env python3
"""
main.py — NS3-AI BSP federated-learning driver

Runs four communication protocols under identical conditions on the same
NS-3 wireless channel:

    dctcp   reliable full-gradient transport (incast-prone)
    ltp     loss-tolerant: Early-Close + Random-K bubble-fill
    plot    per-layer LTT + retx of layers below threshold
    pilt    importance-aware top-|g| mask + error-feedback residual
            + dual-population GA pre-scheduling on the time-frequency
            RB grid

All protocols share the same dataset split, model init, seed, NS-3
instance, K, L and uplink budget — only the transport / aggregation
layer differs.

Outputs   results/metrics_<protocol>.npz   per protocol.

Usage
-----
    python3 main.py --protocols pilt --time_limit 1200 --workers 10
    python3 main.py --protocols dctcp,ltp,plot,pilt --model resnet50 \\
                         --time_limit 1200 --workers 10
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
NS3_DIR  = "/home/goodlab/ns3"
FL_ASYNC = os.path.join(NS3_DIR, "contrib", "ai", "examples", "fl-async")
NS3AI_UTILS_DIR = os.path.join(NS3_DIR, "contrib", "ai", "python_utils")

for p in [FL_ASYNC, NS3AI_UTILS_DIR, os.path.dirname(__file__)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg
from federated.dataset import make_dataset, make_cifar100_dataset, dirichlet_split
from federated.model_torch import FLWorker, ParameterServer, make_model
from protocols.pilt_protocol import (
    PILTProtocol, PILTConfig, GAConfig, WirelessChannelCfg,
)
from protocols.ltp_protocol import LTPProtocol
from protocols.plot_protocol import PLOTProtocol

import ns3ai_fl_py as fl_binding
from ns3ai_utils import Experiment


# ─── Layer grouping (L=10 logical "layer groups" from model parameters) ──────

def group_layers(model: torch.nn.Module, L: int = 10):
    """Chunk model parameters into L roughly-equal groups by element count.

    Returns (groups, group_of_param, group_sizes_elems) where groups[l] is
    the list of parameter indices belonging to logical layer-group l.

    All L groups are guaranteed non-empty when n_params >= L: any empty
    trailing group steals one parameter from the largest non-empty group
    until the partition is full. Total element count is preserved.
    """
    params = list(model.parameters())
    sizes = np.array([p.numel() for p in params], dtype=np.int64)
    n_params = len(params)
    total = int(sizes.sum())
    target_per_group = total / L

    group_of_param = np.zeros(n_params, dtype=np.int64)
    group_sizes_elems = np.zeros(L, dtype=np.int64)
    groups: list[list[int]] = [[] for _ in range(L)]

    g, acc = 0, 0
    for i in range(n_params):
        if g < L - 1 and acc >= target_per_group:
            g += 1
            acc = 0
        groups[g].append(i)
        group_of_param[i] = g
        group_sizes_elems[g] += sizes[i]
        acc += sizes[i]

    # Rebalance: fill any empty groups by stealing one parameter from the
    # largest non-empty group.  Iterate because stealing changes sizes.
    if n_params >= L:
        while True:
            empty = [l for l in range(L) if len(groups[l]) == 0]
            if not empty:
                break
            l_dst = empty[0]
            l_src = int(np.argmax(group_sizes_elems))
            if len(groups[l_src]) <= 1:
                # Cannot drain the source further without emptying it; bail.
                break
            # Move the last param of the largest group → empty group.
            pi = groups[l_src].pop()
            sz = int(sizes[pi])
            group_sizes_elems[l_src] -= sz
            groups[l_dst].append(pi)
            group_of_param[pi] = l_dst
            group_sizes_elems[l_dst] += sz

    return groups, group_of_param, group_sizes_elems


def flat_layer_tensors(grads: list, groups: list, group_sizes_elems: np.ndarray
                       ) -> list[np.ndarray]:
    """Concatenate params within each layer group into one flat np.ndarray.

    Engineering optimisations vs the previous implementation
    (no algorithmic / mathematical change):

      * `.detach().cpu()` was removed — the input tensors are already CPU
        fp32 (FLWorker.compute_gradient returns them on CPU) and have no
        grad attached, so both calls were no-ops that allocated wrappers.
      * `.numpy()` shares storage with the source tensor; we then write
        directly into a single pre-allocated buffer of the exact group
        size, avoiding the two intermediate allocations
        (`.ravel().astype()` + `np.concatenate`) that the previous version
        triggered for every (worker × layer) pair every round.
    """
    out: list[np.ndarray] = []
    for l, g_idx in enumerate(groups):
        size_l = int(group_sizes_elems[l])
        if len(g_idx) == 0 or size_l == 0:
            # Zero-size for empty layer-groups (consistent with
            # group_sizes_elems[l] == 0) — the previous size-1 placeholder
            # silently desynchronised with PILT's residual buffer (which
            # uses group_sizes_elems exactly), causing a downstream
            # boolean-mask shape error in worker_encode.
            out.append(np.zeros(0, dtype=np.float32))
            continue
        flat = np.empty(size_l, dtype=np.float32)
        off = 0
        for i in g_idx:
            t = grads[i]
            arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
            n = arr.size
            flat[off: off + n] = arr.reshape(-1)
            off += n
        out.append(flat)
    return out


def scatter_layer_tensors(layer_flat: list[np.ndarray], groups: list,
                          param_shapes: list, param_sizes: np.ndarray
                          ) -> list[torch.Tensor]:
    """Inverse of flat_layer_tensors.

    Engineering optimisations (no math change):

      * `torch.from_numpy(chunk).view(shape)` instead of
        `torch.tensor(chunk, dtype=torch.float32).view(shape)` — the former
        zero-copies the numpy storage; the latter forces a fresh malloc +
        memcpy of the entire parameter tensor.
      * The chunks are taken as views into the layer-flat buffer
        (`flat[off: off + sz]`) so no per-parameter slice copies are made.
    """
    out: list[torch.Tensor] = [None] * len(param_shapes)
    for l, g_idx in enumerate(groups):
        if len(g_idx) == 0:
            continue
        flat = layer_flat[l]
        if flat.dtype != np.float32:
            flat = flat.astype(np.float32, copy=False)
        if not flat.flags["C_CONTIGUOUS"]:
            flat = np.ascontiguousarray(flat)
        off = 0
        for pi in g_idx:
            sz = int(param_sizes[pi])
            shape = param_shapes[pi]
            chunk = flat[off: off + sz]
            off += sz
            out[pi] = torch.from_numpy(chunk).view(shape)
    for i, t in enumerate(out):
        if t is None:
            out[i] = torch.zeros(param_shapes[i], dtype=torch.float32)
    return out


def apply_packet_damage_flat(flat: np.ndarray, bytes_sent: int,
                              bytes_delivered: int, packet_size: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Zero out a flat vector's positions corresponding to lost packets."""
    if bytes_sent <= 0 or bytes_delivered >= bytes_sent:
        return flat
    fpkt = max(1, packet_size // 4)
    n_pkts = max(1, (flat.size * 4 + packet_size - 1) // packet_size)
    loss = max(0.0, 1.0 - bytes_delivered / bytes_sent)
    lost = rng.random(n_pkts) < loss
    mask = np.ones(flat.size, dtype=bool)
    for i in range(n_pkts):
        if lost[i]:
            lo = i * fpkt
            hi = min(lo + fpkt, flat.size)
            mask[lo:hi] = False
    return np.where(mask, flat, np.float32(0.0))


# ─── Simulation core ─────────────────────────────────────────────────────────

def run_simulation(protocol_name   : str,
                   X_train         : np.ndarray,
                   y_train         : np.ndarray,
                   X_test          : np.ndarray,
                   y_test          : np.ndarray,
                   worker_indices  : list,
                   max_rounds      : int,
                   time_limit_ms   : float,
                   seed            : int,
                   msg_interface,
                   backbone_mbps   : float,
                   per_worker_uplink_mbps : float,
                   L_groups        : int = 10,
                   pilt_E_total    : float = 0.5,
                   pilt_d          : float = 0.05,
                   pilt_n_rbs      : int = 10,
                   pilt_rb_bw_khz  : float = 900.0,
                   pilt_pipeline_ga: bool = True,
                   verbose_every   : int = 5,
                   checkpoint_path : str | None = None,
                   checkpoint_every: int = 50,
                  ) -> dict:
    """One protocol run. Returns metrics dictionary."""

    n_workers = len(worker_indices)
    torch.manual_seed(seed); np.random.seed(seed)

    # ── Build global model + PS + workers ───────────────────────────────────
    # One "scratch" model is allocated on GPU and *shared* across all workers.
    # Workers never hold their own GPU-resident model — this keeps peak GPU
    # memory at 1×(model + activations) instead of K×, which is essential for
    # ResNet-50/VGG-16 when the GPU is shared with other processes.
    global_model = make_model(cfg.MODEL)
    ps = ParameterServer(cfg.MODEL, X_test, y_test)
    ps.model.load_state_dict(global_model.state_dict())

    scratch_model = make_model(cfg.MODEL).to(ps.model.weight.device
                                              if hasattr(ps.model, "weight")
                                              else next(ps.model.parameters()).device)

    workers: list[FLWorker] = []
    rng = np.random.RandomState(seed)
    for wid, idx in enumerate(worker_indices):
        w = FLWorker(
            worker_id = wid,
            X_local   = X_train[idx], y_local = y_train[idx],
            model_cfg = cfg.MODEL, sim_cfg = cfg.SIMULATION,
            rng       = np.random.RandomState(rng.randint(0, 2**31)),
        )
        w.attach_shared_model(scratch_model)
        workers.append(w)

    # ── Layer grouping ───────────────────────────────────────────────────────
    groups, group_of_param, group_sizes_elems = group_layers(
        global_model, L=L_groups)
    L = len(groups)

    param_shapes = [tuple(p.shape) for p in global_model.parameters()]
    param_sizes  = np.array([p.numel() for p in global_model.parameters()],
                            dtype=np.int64)

    # Rough per-layer compute fraction (proxy: elements in group / total)
    total_elems = float(group_sizes_elems.sum())
    layer_cfrac = np.cumsum(group_sizes_elems) / max(1.0, total_elems)

    # ── Protocol instances ──────────────────────────────────────────────────
    pilt_obj: PILTProtocol | None = None
    ltp_obj : LTPProtocol  | None = None
    plot_obj: PLOTProtocol | None = None
    if protocol_name == "pilt":
        # Default discretisation: M=10 RBs × 900 kHz ≈ 9 MHz total wireless,
        # Δτ = 1 ms. Coarser RBs cut the GA inner loop ~5× vs the LTE-style
        # 50 × 180 kHz grid; total bandwidth is preserved.
        ga_cfg = GAConfig(Np=50, G_max=100, seed=seed + 7)
        ch_cfg = WirelessChannelCfg(
            bandwidth_per_rb_hz = float(pilt_rb_bw_khz) * 1e3,
            n_rbs               = int(pilt_n_rbs),
            slot_duration_ms    = 1.0,
        )
        pilt_cfg = PILTConfig(
            beta=0.9, d=pilt_d, E_total=pilt_E_total, eps_min=0.05,
            ga=ga_cfg, ch=ch_cfg,
        )
        pilt_obj = PILTProtocol(
            n_workers=n_workers, n_layers=L,
            layer_sizes=group_sizes_elems.tolist(),
            layer_compute_fraction=layer_cfrac,
            dataset_sizes_per_worker=[cfg.MODEL["batch_size"]] * n_workers,
            epochs_E=cfg.MODEL.get("local_steps", 1),
            per_sample_compute_ms=0.02,
            cfg=pilt_cfg,
        )
    elif protocol_name == "ltp":
        # RTprop ≈ 2·one-way delay; BtlBw ≈ min(uplink, backbone/n_workers)
        rtprop_ms  = 2.0
        btlbw_mbps = min(per_worker_uplink_mbps, backbone_mbps / n_workers)
        ltp_obj = LTPProtocol(
            n_workers, L, group_sizes_elems.tolist(),
            rtprop_ms=rtprop_ms, btlbw_mbps=btlbw_mbps,
            deadline_slack_ms=30.0,
            min_percent=0.80,
            seed=seed + 3003,
        )
    elif protocol_name == "plot":
        from protocols.plot_protocol import (
            layer_ltt_resnet, default_resnet20_structure,
        )
        bb, rc = default_resnet20_structure(L)
        layer_ltt = layer_ltt_resnet(
            group_sizes_elems.tolist(), bb, rc,
            total_ltt=0.20,
            d_block_factor=0.8,
            rc_ratio=0.5,
        )
        plot_obj = PLOTProtocol(
            n_workers, L, group_sizes_elems.tolist(),
            layer_ltt=layer_ltt,
            seed=seed + 4004,
        )

    # ── NS3 handle ──────────────────────────────────────────────────────────
    req_vec = msg_interface.GetPy2CppVector()
    res_vec = msg_interface.GetCpp2PyVector()
    pkt_sz  = cfg.NETWORK["packet_size_bytes"]

    # ── Metrics ─────────────────────────────────────────────────────────────
    wall_ms = 0.0
    history = {
        "wall_ms"     : [],
        "test_acc"    : [],
        "test_loss"   : [],
        "round_ms"    : [],
        "per_worker_fct_ms": [],          # list[list]
        "bytes_sent_total": 0,
        "bytes_delivered_total": 0,
        "makespan_slots"  : [],           # PILT only
        # PILT GA cost: intrinsic (raw GA wall-time on the PS host) is
        # always added into per-round wall-clock; ps_exposed_ms is what
        # remains after pipelining with worker SGD = max(0, GA - max_k T_cmp).
        # Both stay zero for LTP / PLOT / DCTCP.
        "ps_overhead_ms"  : [],
        "ps_exposed_ms"   : [],
    }
    test_acc = 1.0 / cfg.DATASET["n_classes"]
    test_loss = np.log(cfg.DATASET["n_classes"])

    pilt_rng = np.random.default_rng(seed + 1001)
    damage_rng = np.random.default_rng(seed + 2002)

    display_name = {
        "dctcp": "DCTCP", "ltp": "LTP", "plot": "PLOT", "pilt": "PILT",
    }[protocol_name]

    print(f"\n{'=' * 72}")
    print(f"  Protocol: {display_name:<6}  "
          f"K={n_workers}  L={L}  "
          f"backbone={backbone_mbps:.0f}Mbps  "
          f"uplink={per_worker_uplink_mbps:.0f}Mbps "
          f"time_limit={time_limit_ms/1000:.0f}s")
    print(f"{'=' * 72}")

    start_wall = time.time()
    r = 0
    # ── GA pipelining executor ─────────────────────────────────────────────
    # The GA solver is pure Python+NumPy; numpy hot loops release the GIL,
    # so a background thread overlaps cleanly with PyTorch CUDA work in the
    # main thread (CUDA kernels also release the GIL while waiting).  We
    # always create the executor so the cleanup path is uniform; non-PILT
    # protocols simply never submit jobs to it.  Shut it down at the end of
    # `run_simulation` (see below).
    ga_executor = ThreadPoolExecutor(max_workers=1,
                                     thread_name_prefix="pilt-ga")
    while wall_ms < time_limit_ms and r < max_rounds:
        # ── 1. Wall-clock LR schedule (fair across protocols) ───────────────
        # Linear warm-up for the first `warmup_frac` of training, then
        # piecewise decay at 0.6T / 0.85T.  Applied identically to every
        # protocol so the comparison is fair.
        base_lr = cfg.MODEL["lr"]
        warmup_frac = float(cfg.MODEL.get("warmup_frac", 0.0))
        prog = wall_ms / time_limit_ms
        if warmup_frac > 0.0 and prog < warmup_frac:
            # avoid lr=0 on round 1
            lr = base_lr * max(1e-3, prog / warmup_frac)
        elif prog < 0.6:
            lr = base_lr
        elif prog < 0.85:
            lr = base_lr * 0.1
        else:
            lr = base_lr * 0.01
        for w in workers: w.set_lr(lr)

        # ── 2. Kick off PILT GA in background (overlaps with worker SGD) ────
        # GA inputs depend only on the importance EMA from the previous
        # round and per-worker compute times stored inside the protocol,
        # so the solver can run in a thread alongside worker SGD. The
        # main thread joins on the future before encoding.
        ga_future = None
        ga_eps    = None
        ga_D      = None
        if protocol_name == "pilt" and pilt_pipeline_ga:
            assert pilt_obj is not None
            ga_eps = pilt_obj.compute_ratios()       # microseconds
            ga_D   = pilt_obj.target_payload_bytes(ga_eps)  # microseconds
            def _ga_solve_with_timing(_pilt=pilt_obj, _D=ga_D):
                _t0 = time.perf_counter()
                _sched, _mk = _pilt.solve_schedule(_D)
                return _sched, _mk, (time.perf_counter() - _t0) * 1000.0
            ga_future = ga_executor.submit(_ga_solve_with_timing)

        # ── 3. Compute per-worker gradients ─────────────────────────────────
        # Snapshot the global weights once per round; all K workers reuse
        # the same snapshot at the start of their SGD (BSP semantics),
        # which avoids K-1 full GPU model clones per round.
        global_model_ref = ps.get_global_model()
        global_state = global_model_ref.state_dict()
        shared_snap = [p.data.detach().clone()
                       for p in global_model_ref.parameters()]
        grads_all_params: list = []     # list of list[Tensor] (per-param)
        comp_ms: list[float] = []
        for w in workers:
            grads, _grad_bytes, cmp_ms = w.compute_gradient(
                global_state, initial_params=shared_snap)
            grads_all_params.append(grads)
            comp_ms.append(cmp_ms)
        del global_state, shared_snap

        # ── 4. Group + protocol encode ─────────────────────────────────────
        grads_by_layer_per_worker: list[list[np.ndarray]] = [
            flat_layer_tensors(gp, groups, group_sizes_elems)
            for gp in grads_all_params
        ]

        encoded_per_worker: list[list[np.ndarray]] = []    # to aggregate
        mask_per_worker   : list[list[np.ndarray]] = []
        bytes_total_per_worker: list[int] = []
        bytes_per_layer_per_worker: list[np.ndarray] = []

        # PS-side per-round overhead (real wall-clock seconds). Only PILT
        # produces non-zero overhead here. Measured from the actual Python
        # time of importance-rank → ratio → GA solve, then added into round_ms.
        ps_overhead_ms = 0.0

        if protocol_name == "pilt":
            assert pilt_obj is not None
            if ga_future is not None:
                # Pipelined path — GA was kicked off before worker SGD.
                # Reuse the ε_l / D we already computed; only the heavy
                # `solve_schedule` ran in the background.
                eps = ga_eps
                D   = ga_D
                sched, makespan_slots, ga_intrinsic_ms = ga_future.result()
                ps_overhead_ms = ga_intrinsic_ms
            else:
                _t_pilt_pre = time.perf_counter()
                eps = pilt_obj.compute_ratios()
                D = pilt_obj.target_payload_bytes(eps)
                sched, makespan_slots = pilt_obj.solve_schedule(D)
                ps_overhead_ms = (time.perf_counter() - _t_pilt_pre) * 1000.0
            history["makespan_slots"].append(makespan_slots)

            for k in range(n_workers):
                eff, mask, _sent = pilt_obj.worker_encode(
                    k, grads_by_layer_per_worker[k], eps)
                encoded_per_worker.append(eff)
                mask_per_worker.append(mask)
                bytes_per_layer_per_worker.append(D[k].astype(np.int64))
                bytes_total_per_worker.append(int(D[k].sum()))

            start_slots, _ = pilt_obj.schedule_to_worker_plan(sched, eps)
            # Convert slots (1ms each by default) to ms
            start_times_ms = start_slots.astype(np.float32) * \
                             float(pilt_obj.cfg.ch.slot_duration_ms)
            proto_id = 2
        elif protocol_name == "ltp":
            # LTP: worker sends the FULL gradient on a lossy channel;
            # loss is decided later by Early-Close after NS3 reports FCT.
            D = ltp_obj.target_payload_bytes()
            for k in range(n_workers):
                eff = [np.asarray(g).ravel().astype(np.float32, copy=False)
                       for g in grads_by_layer_per_worker[k]]
                encoded_per_worker.append(eff)
                mask_per_worker.append([np.ones(g.size, dtype=bool) for g in eff])
                bytes_per_layer_per_worker.append(D[k].astype(np.int64))
                bytes_total_per_worker.append(int(D[k].sum()))
            start_times_ms = np.zeros(n_workers, dtype=np.float32)
            proto_id = 1
        elif protocol_name == "plot":
            # PLOT: send full gradient on AUDP; PS decides per-layer retx
            # after observing loss. Second NS3 pass happens below if any
            # layer exceeds its Layer-LTT.
            D = plot_obj.target_payload_bytes()
            for k in range(n_workers):
                eff = [np.asarray(g).ravel().astype(np.float32, copy=False)
                       for g in grads_by_layer_per_worker[k]]
                encoded_per_worker.append(eff)
                mask_per_worker.append([np.ones(g.size, dtype=bool) for g in eff])
                bytes_per_layer_per_worker.append(D[k].astype(np.int64))
                bytes_total_per_worker.append(int(D[k].sum()))
            start_times_ms = np.zeros(n_workers, dtype=np.float32)
            proto_id = 1
        else:  # dctcp — reliable, ECN-style
            for k in range(n_workers):
                eff = [g.astype(np.float32) for g in grads_by_layer_per_worker[k]]
                encoded_per_worker.append(eff)
                mask_per_worker.append([np.ones(g.size, dtype=bool) for g in eff])
                bpl = np.array([g.size * 4 for g in eff], dtype=np.int64)
                bytes_per_layer_per_worker.append(bpl)
                bytes_total_per_worker.append(int(bpl.sum()))
            start_times_ms = np.zeros(n_workers, dtype=np.float32)
            proto_id = 0

        # ── 5. Push to NS3 ──────────────────────────────────────────────────
        msg_interface.PySendBegin()
        for k in range(n_workers):
            req_vec[k].worker_id      = k
            req_vec[k].bytes_to_send  = max(1, int(bytes_total_per_worker[k]))
            req_vec[k].bandwidth_mbps = float(per_worker_uplink_mbps)
            req_vec[k].protocol       = proto_id
            req_vec[k].round_num      = r
            req_vec[k].start_time_ms  = float(start_times_ms[k])
            req_vec[k].backbone_mbps  = float(backbone_mbps)
            req_vec[k].lossless       = 0   # BSP path keeps env loss model
        msg_interface.PySendEnd()

        # ── 6. Receive NS3 results ──────────────────────────────────────────
        msg_interface.PyRecvBegin()
        ns3_out = [(int(res_vec[k].bytes_delivered),
                    float(res_vec[k].delay_ms),
                    int(res_vec[k].pkts_sent),
                    int(res_vec[k].pkts_received))
                   for k in range(n_workers)]
        msg_interface.PyRecvEnd()

        # Debug: log per-worker delivery ratio every 10 rounds
        if os.environ.get("PILT_DEBUG_NS3") and (r < 3 or (r + 1) % 10 == 0):
            ratios = [ns3_out[k][0] / max(1, bytes_total_per_worker[k])
                      for k in range(n_workers)]
            print(f"  [ns3 R{r+1}] proto={proto_id} "
                  f"ratio min/avg/max = {min(ratios):.3f}/"
                  f"{sum(ratios)/len(ratios):.3f}/{max(ratios):.3f} "
                  f"pkt_delivered/sent = "
                  f"{sum(v[3] for v in ns3_out)}/{sum(v[2] for v in ns3_out)}",
                  flush=True)

        fct_list = [v[1] for v in ns3_out]     # per-worker FCT (ms)

        # ── 7. Protocol-specific delivery semantics ─────────────────────────
        #
        # Strategy:
        # * DCTCP / PILT — simple flow-level loss; bubble-fill uniformly.
        # * LTP          — Early-Close: truncate the flow at LT-threshold /
        #                   deadline then bubble-fill the resulting
        #                   (Random-k) remainder.
        # * PLOT         — detect per-layer loss; if any layer exceeds its
        #                   Layer-LTT, make a 2nd NS3 call with retx-only
        #                   bytes and merge.
        # ----------------------------------------------------------------------

        # Per-worker per-layer delivered fraction ∈ [0,1]
        delivered_frac_per_layer = np.ones((n_workers, L), dtype=np.float64)

        if protocol_name == "ltp":
            for k in range(n_workers):
                b_sent  = int(bytes_total_per_worker[k])
                b_delvd = int(ns3_out[k][0])
                full_fct = float(fct_list[k])
                # Proportional arrival rate (constant bitrate model)
                lt   = ltp_obj.lt_threshold_ms(k)
                dl   = ltp_obj.deadline_ms(k)
                full_completion = full_fct if b_delvd >= b_sent else None
                # Fraction arrived by time t = min(1, b_delvd * t / (b_sent * full_fct))
                # Simplify: if flow needs full_fct ms for b_delvd bytes,
                # rate = b_delvd / full_fct.  Fraction of total bytes at t:
                #   arrived(t) = min(b_delvd, rate·t) / b_sent
                def arrived_fraction(t_ms: float) -> float:
                    if full_fct <= 0:
                        return 1.0
                    delivered_at_t = min(b_delvd, (b_delvd / full_fct) * t_ms)
                    return max(0.0, min(1.0, delivered_at_t / max(1, b_sent)))
                at_lt = arrived_fraction(lt)
                at_dl = arrived_fraction(dl)
                frac, reason = ltp_obj.early_close_decision(
                    k, at_lt, at_dl, full_completion
                )
                delivered_frac_per_layer[k, :] = frac
                # Effective FCT for BSP = time when PS closes the flow
                if reason == "full":
                    fct_list[k] = full_fct
                elif reason == "early":
                    # close as soon as LT-threshold is met
                    fct_list[k] = lt if at_lt >= ltp_obj._ec[k].min_percent else dl
                else:  # deadline
                    fct_list[k] = dl

        elif protocol_name == "plot":
            for k in range(n_workers):
                b_sent  = int(bytes_total_per_worker[k])
                b_delvd = int(ns3_out[k][0])
                ratio   = max(0.0, min(1.0, b_delvd / max(1, b_sent)))
                delivered_frac_per_layer[k, :] = ratio

            eff_plot, mask_plot = plot_obj.simulate_delivery(
                grads_by_layer_per_worker, delivered_frac_per_layer,
            )
            retx_layers = plot_obj.retx_layers_per_worker(
                delivered_frac_per_layer
            )
            retx_bytes = plot_obj.retx_payload_bytes()

            if retx_bytes is not None and retx_bytes.sum() > 0:
                # Second NS3 call: retx-only bytes for layers under their LTT.
                msg_interface.PySendBegin()
                for k in range(n_workers):
                    req_vec[k].worker_id      = k
                    req_vec[k].bytes_to_send  = max(1, int(retx_bytes[k]))
                    req_vec[k].bandwidth_mbps = float(per_worker_uplink_mbps)
                    req_vec[k].protocol       = 1                 # AUDP
                    req_vec[k].round_num      = r
                    req_vec[k].start_time_ms  = 0.0
                    req_vec[k].backbone_mbps  = float(backbone_mbps)
                    req_vec[k].lossless       = 0
                msg_interface.PySendEnd()
                msg_interface.PyRecvBegin()
                retx_out = [(int(res_vec[k].bytes_delivered),
                             float(res_vec[k].delay_ms))
                            for k in range(n_workers)]
                msg_interface.PyRecvEnd()

                retx_frac = np.zeros((n_workers, L), dtype=np.float64)
                for k in range(n_workers):
                    rbs = int(retx_bytes[k])
                    if rbs <= 0:
                        continue
                    rbd = int(retx_out[k][0])
                    ratio = max(0.0, min(1.0, rbd / max(1, rbs)))
                    for l in retx_layers[k]:
                        retx_frac[k, l] = ratio
                    # Extend the round FCT
                    fct_list[k] = fct_list[k] + float(retx_out[k][1])
                    history["bytes_sent_total"]      += rbs
                    history["bytes_delivered_total"] += rbd

                eff_plot, mask_plot = plot_obj.apply_retx_delivery(
                    grads_by_layer_per_worker, eff_plot, mask_plot,
                    retx_layers, retx_frac,
                )

            # Stash PLOT's post-delivery grads into encoded_per_worker so the
            # common rescatter path below works unchanged.
            encoded_per_worker = eff_plot
            mask_per_worker    = mask_plot

        # BSP round time. PILT GA can run in parallel with local SGD on
        # workers; worker k can only start its uplink once both its local
        # SGD AND the GA schedule are ready, i.e. at max(T_GA, T_cmp_k).
        # Round end is then max_k ( max(T_GA, T_cmp_k) + T_tx_k ).
        # When pipelining is off we recover  T_GA + max_k(T_cmp_k + T_tx_k).
        # ps_overhead_ms = intrinsic GA time;
        # ps_exposed_ms  = the part that surfaces past compute mask.
        if protocol_name == "pilt" and pilt_pipeline_ga:
            round_ms = max(
                max(ps_overhead_ms, comp_ms[k]) + fct_list[k]
                for k in range(n_workers)
            )
            ps_exposed_ms = max(
                0.0, ps_overhead_ms - max(comp_ms)
            )
        else:
            worker_max_ms = max(comp_ms[k] + fct_list[k]
                                for k in range(n_workers))
            round_ms = ps_overhead_ms + worker_max_ms
            ps_exposed_ms = ps_overhead_ms
        wall_ms += round_ms

        # ── Bubble-fill / scatter → per-parameter tensors ────────────────────
        recovered_per_worker: list[list[torch.Tensor]] = []
        for k in range(n_workers):
            b_sent  = int(bytes_total_per_worker[k])
            b_delvd = int(ns3_out[k][0])

            damaged_layers_flat: list[np.ndarray] = []
            if protocol_name == "ltp":
                # All layers share the same Early-Close fraction this round.
                frac = float(delivered_frac_per_layer[k, 0])
                eff, _m, _s = ltp_obj.bubble_fill(
                    k, grads_by_layer_per_worker[k], frac
                )
                damaged_layers_flat = eff
            elif protocol_name == "plot":
                # Already bubble-filled by PLOT (with possible retx merge).
                damaged_layers_flat = encoded_per_worker[k]
            else:
                # DCTCP / PILT — flow-level ratio applied uniformly.
                bpl = bytes_per_layer_per_worker[k]
                ratio = max(0.0, min(1.0, b_delvd / max(1, b_sent)))
                for l, eff_l in enumerate(encoded_per_worker[k]):
                    b_l_delvd = int(bpl[l] * ratio)
                    damaged = apply_packet_damage_flat(
                        eff_l, int(bpl[l]), b_l_delvd, pkt_sz, damage_rng
                    )
                    damaged_layers_flat.append(damaged)

            # Re-scatter layer-flat back to per-parameter tensors
            rec_tensors = scatter_layer_tensors(
                damaged_layers_flat, groups, param_shapes, param_sizes)
            recovered_per_worker.append(rec_tensors)

            history["bytes_sent_total"]      += b_sent
            history["bytes_delivered_total"] += b_delvd

        # ── 8. PS aggregation — FedAvg with equal weights (BSP semantics) ───
        weights = [1.0] * n_workers
        ps.aggregate(recovered_per_worker, weights)

        # LTP: end-of-epoch threshold update
        # We use "every N rounds" as a pseudo-epoch (cifar epoch ≈ 50 rounds
        # for K=10 batch 32 on 50k samples).
        if protocol_name == "ltp" and (r + 1) % 50 == 0:
            ltp_obj.end_epoch()

        # ── 9. PILT importance update (post-aggregate) ─────────────────────
        if protocol_name == "pilt":
            # In-place running sum to compute per-layer mean across K
            # workers — avoids allocating a (K, S_l) stack per layer.
            avg_layers_np: list[np.ndarray] = []
            inv_K = 1.0 / float(n_workers)
            for l in range(L):
                acc = encoded_per_worker[0][l].astype(np.float32, copy=True)
                for k in range(1, n_workers):
                    acc += encoded_per_worker[k][l]
                acc *= inv_K
                avg_layers_np.append(acc)
            pilt_obj.update_importance(avg_layers_np)

        # ── 10. Evaluate ───────────────────────────────────────────────────
        EVAL_EVERY = 5
        if r == 0 or (r + 1) % EVAL_EVERY == 0:
            test_loss, test_acc = ps.evaluate()

        history["wall_ms"].append(wall_ms)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["round_ms"].append(round_ms)
        history["per_worker_fct_ms"].append(fct_list)
        history["ps_overhead_ms"].append(ps_overhead_ms)
        history["ps_exposed_ms"].append(
            ps_exposed_ms if protocol_name == "pilt" else 0.0)

        if r == 0 or (r + 1) % verbose_every == 0:
            real_s = time.time() - start_wall
            extra = ""
            if protocol_name == "pilt":
                extra = (f" | makespan={history['makespan_slots'][-1]}slots"
                         f" | ga={ps_overhead_ms:>6.1f}ms"
                         f" exp={ps_exposed_ms:>5.1f}ms")
            print(f"  R{r+1:>4} | sim={wall_ms/1000:>7.2f}s | "
                  f"acc={test_acc:.3f} | loss={test_loss:.3f} | "
                  f"roundT={round_ms:>6.1f}ms | real={real_s:>5.1f}s{extra}",
                  flush=True)

        # Periodic checkpoint to disk for long runs.
        if checkpoint_path is not None and (r + 1) % checkpoint_every == 0:
            _ckpt = {
                "protocol" : protocol_name,
                "wall_ms"  : np.array(history["wall_ms"]),
                "test_acc" : np.array(history["test_acc"]),
                "test_loss": np.array(history["test_loss"]),
                "round_ms" : np.array(history["round_ms"]),
                "rounds"   : np.arange(1, r + 2),
                "per_worker_fct_ms": np.array(history["per_worker_fct_ms"]),
                "ps_overhead_ms"   : np.array(history["ps_overhead_ms"]),
                "ps_exposed_ms"    : np.array(history["ps_exposed_ms"]),
                "bytes_sent_total"     : history["bytes_sent_total"],
                "bytes_delivered_total": history["bytes_delivered_total"],
            }
            if history["makespan_slots"]:
                _ckpt["makespan_slots"] = np.array(history["makespan_slots"])
            try:
                np.savez(checkpoint_path,
                         **{k: v for k, v in _ckpt.items()
                            if isinstance(v, np.ndarray)})
            except Exception as _e:
                print(f"  [ckpt warn] {_e}", flush=True)

        r += 1

    # Tear down GA executor before returning.  `wait=False` because nothing
    # is in-flight at this point — the last round's `.result()` already
    # returned before the loop exited.
    ga_executor.shutdown(wait=False)

    out = {
        "protocol" : protocol_name,
        "wall_ms"  : np.array(history["wall_ms"]),
        "test_acc" : np.array(history["test_acc"]),
        "test_loss": np.array(history["test_loss"]),
        "round_ms" : np.array(history["round_ms"]),
        "rounds"   : np.arange(1, r + 1),
        "per_worker_fct_ms": np.array(history["per_worker_fct_ms"]),
        "ps_overhead_ms"   : np.array(history["ps_overhead_ms"]),
        "ps_exposed_ms"    : np.array(history["ps_exposed_ms"]),
        "bytes_sent_total"     : history["bytes_sent_total"],
        "bytes_delivered_total": history["bytes_delivered_total"],
    }
    if history["makespan_slots"]:
        out["makespan_slots"] = np.array(history["makespan_slots"])
    return out


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NS3-AI BSP federated-learning driver: DCTCP / LTP / "
                    "PLOT / PILT under identical conditions.")
    ap.add_argument("--protocol",
                    choices=["dctcp", "ltp", "plot", "pilt", "all"],
                    default="all",
                    help="which transport to run (default: all four). "
                         "For an arbitrary subset use --protocols.")
    ap.add_argument("--protocols", default=None,
                    help="comma-separated subset (overrides --protocol). "
                         "Example: --protocols dctcp,ltp,pilt")
    ap.add_argument("--workers", type=int, default=10,
                    help="K — number of FL workers")
    ap.add_argument("--rounds", type=int, default=1_000_000,
                    help="hard cap on BSP rounds (time_limit usually binds first)")
    ap.add_argument("--time_limit", type=float, default=40000.0,
                    help="simulated training seconds per protocol")
    ap.add_argument("--model", choices=["resnet20", "resnet50", "vgg16"],
                    default=None,
                    help="override config.py MODEL.type")
    ap.add_argument("--seed", type=int, default=cfg.SEED,
                    help="master seed used for dataset split, model init, "
                         "channel and GA — identical across all protocols")
    ap.add_argument("--layer_groups", type=int, default=10,
                    help="L — number of layer groups")
    ap.add_argument("--backbone_mbps", type=float, default=10000.0,
                    help="router→PS bottleneck bandwidth")
    ap.add_argument("--uplink_mbps", type=float, default=200.0,
                    help="per-worker wireless uplink capacity (Mbps)")
    ap.add_argument("--pilt_E_total", type=float, default=0.5,
                    help="global transmission budget E_total ∈ (0,1]")
    ap.add_argument("--pilt_d", type=float, default=0.05,
                    help="rank-update step d for ε_l")
    ap.add_argument("--pilt_n_rbs", type=int, default=10,
                    help="M — number of frequency RBs in the GA grid. "
                         "Pass --pilt_n_rbs 50 --pilt_rb_bw_khz 180 to use "
                         "the LTE-style 50×180 kHz discretisation.")
    ap.add_argument("--pilt_rb_bw_khz", type=float, default=900.0,
                    help="W — per-RB bandwidth in kHz (default 900 kHz so "
                         "M·W = 9 MHz total).")
    ap.add_argument("--pilt_no_pipeline_ga", action="store_true",
                    help="Disable GA / worker-SGD pipelining (ablation).")
    args = ap.parse_args()

    # Optional model override propagates into the shared config used below
    if args.model is not None:
        cfg.MODEL["type"] = args.model

    # ── Dataset ─────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    dataset_type = cfg.DATASET.get("type", "synthetic")
    if dataset_type == "cifar100":
        print("Loading CIFAR-100 …", flush=True)
        X_tr, y_tr, X_te, y_te = make_cifar100_dataset(
            data_root=cfg.DATASET.get("data_root", "./data"), rng=rng)
    else:
        X_tr, y_tr, X_te, y_te = make_dataset(
            n_samples  = cfg.DATASET["n_samples"],
            n_test     = cfg.DATASET["n_test"],
            n_features = cfg.DATASET["n_features"],
            n_classes  = cfg.DATASET["n_classes"],
            noise      = cfg.DATASET.get("noise", 0.15),
            rng        = rng,
        )

    worker_indices = dirichlet_split(
        y=y_tr, n_workers=args.workers,
        alpha=cfg.DATASET["non_iid_alpha"], rng=rng,
    )

    n_params = sum(p.numel() for p in make_model(cfg.MODEL).parameters())
    print(f"\nDataset : {dataset_type.upper()}  "
          f"{len(X_tr)} train / {len(X_te)} test  |  "
          f"{args.workers} workers (non-IID α={cfg.DATASET['non_iid_alpha']})")
    print(f"Model   : {cfg.MODEL['type'].upper()}  "
          f"{n_params:,} params ({n_params*4/1024:.0f} KB gradient)  "
          f"lr={cfg.MODEL['lr']}")
    print(f"Network : backbone={args.backbone_mbps:.0f} Mbps "
          f"(incast bottleneck)  |  uplink={args.uplink_mbps:.0f} Mbps/worker")

    if args.protocols is not None:
        protos = [p.strip() for p in args.protocols.split(",") if p.strip()]
        valid = {"dctcp", "ltp", "plot", "pilt"}
        bad = [p for p in protos if p not in valid]
        if bad:
            ap.error(f"unknown protocol(s) in --protocols: {bad}")
    elif args.protocol == "all":
        protos = ["dctcp", "ltp", "plot", "pilt"]
    else:
        protos = [args.protocol]
    print(f"\nRunning protocols: {', '.join(protos)}  "
          f"({args.time_limit:.0f}s each, up to {args.rounds} rounds)")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results")
    os.makedirs(results_dir, exist_ok=True)

    exp = Experiment(
        targetName   = "ns3ai_fl_sim",
        ns3Path      = NS3_DIR,
        msgModule    = fl_binding,
        handleFinish = False,
        useVector    = True,
        vectorSize   = args.workers,
        shmSize      = 65536,
    )
    msg_interface = exp.run(show_output=False)

    all_metrics = {}
    try:
        for proto in protos:
            ckpt_path = os.path.join(results_dir, f"metrics_{proto}.npz")
            metrics = run_simulation(
                protocol_name = proto,
                X_train       = X_tr,  y_train = y_tr,
                X_test        = X_te,  y_test  = y_te,
                worker_indices = worker_indices,
                max_rounds    = args.rounds,
                time_limit_ms = args.time_limit * 1000.0,
                seed          = args.seed,
                msg_interface = msg_interface,
                backbone_mbps = args.backbone_mbps,
                per_worker_uplink_mbps = args.uplink_mbps,
                L_groups      = args.layer_groups,
                pilt_E_total  = args.pilt_E_total,
                pilt_d        = args.pilt_d,
                pilt_n_rbs    = args.pilt_n_rbs,
                pilt_rb_bw_khz= args.pilt_rb_bw_khz,
                pilt_pipeline_ga = not args.pilt_no_pipeline_ga,
                checkpoint_path = ckpt_path,
                checkpoint_every = 50,
            )
            all_metrics[proto] = metrics
            out_path = os.path.join(results_dir, f"metrics_{proto}.npz")
            np.savez(out_path,
                     **{k: v for k, v in metrics.items()
                        if isinstance(v, np.ndarray)})
            print(f"  → saved {out_path}")

    except Exception:
        traceback.print_exc()
    finally:
        try:
            msg_interface.PySendBegin()
            req_vec = msg_interface.GetPy2CppVector()
            req_vec[0].bytes_to_send = 0
            msg_interface.PySendEnd()
        except Exception:
            pass
        del exp

    # ── Summary ─────────────────────────────────────────────────────────────
    if all_metrics:
        print(f"\n{'─'*104}")
        print(f"  {'Protocol':<10} {'Rounds':>8} {'Wall(s)':>10} "
              f"{'Final Acc':>12} {'Mean RoundMs':>14} {'p99 FCT ms':>12} "
              f"{'GA(intr)ms':>12} {'GA(exp)ms':>12}")
        print(f"{'─'*104}")
        base_jct = None
        for proto, m in all_metrics.items():
            wall_s = m["wall_ms"][-1] / 1000.0
            rds    = len(m["rounds"])
            acc    = m["test_acc"][-1]
            mean_r = float(m["round_ms"].mean())
            fcts   = m["per_worker_fct_ms"].ravel()
            p99    = float(np.quantile(fcts, 0.99)) if fcts.size else 0.0
            ps_ovh = float(m["ps_overhead_ms"].mean()) \
                     if "ps_overhead_ms" in m and m["ps_overhead_ms"].size else 0.0
            ps_exp = float(m["ps_exposed_ms"].mean()) \
                     if "ps_exposed_ms" in m and m["ps_exposed_ms"].size else 0.0
            print(f"  {proto.upper():<10} {rds:>8} {wall_s:>10.1f}   "
                  f"{acc:>10.3f}   {mean_r:>12.1f}   {p99:>10.1f}   "
                  f"{ps_ovh:>10.1f}   {ps_exp:>10.1f}")
            if proto == "dctcp":
                base_jct = wall_s
        print(f"{'─'*104}")
        print("  GA(intr) = intrinsic GA wall-time / round (charged in "
              "deployment too); GA(exp) = portion that surfaces past the "
              "worker-SGD mask after pipelining (effective contribution to "
              "Wall(s)).  Both 0 for non-PILT protocols.")

        if base_jct is not None:
            print("\nNormalised JCT (relative to DCTCP):")
            for proto, m in all_metrics.items():
                rel = (m["wall_ms"][-1] / 1000.0) / base_jct
                print(f"    {proto.upper():<6} = {rel:.3f}")

        print("\nRun  python plot_results.py  to generate comparison figures.\n")


if __name__ == "__main__":
    main()
