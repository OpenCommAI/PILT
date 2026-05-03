"""protocols/autotuner.py — built-in overlap adjudicator (内置判决模块).

A startup micro-benchmark ("内测") that probes the actual per-round
worker-SGD wall-time T_cmp and the dual-population GA solver wall-time
T_GA on the live problem instance, then picks

    * dual-population GA hyper-parameters  (Np, G_max)
    * worker local-step count              (E = local_steps)

so that the GA scheduler's intrinsic wall-time is fully masked by
worker SGD when GA / SGD pipelining is enabled.  When this is achieved,
``ps_exposed_ms → 0`` and the algorithm overhead is "hidden in the
computation process" — the user never pays it on the round critical
path.

This module is **opt-in**.  PILT runs unchanged when the master switch
``AutoTuneConfig.enabled`` is False (the default).  In ``main.py`` the
toggle is exposed as ``--autotune``.

Public surface
--------------
    AutoTuneConfig   — public knobs
    AutoTuneReport   — what was probed and what was applied
    OverlapAutoTuner — single-shot calibrator with a `.calibrate(...)`
                       method that mutates the live PILTProtocol /
                       FLWorker / model+sim configs in place.

The mutation is restricted to fields the live runtime reads
fresh each round:

    * ``pilt_obj.cfg.ga.Np`` / ``pilt_obj.cfg.ga.G_max``
    * ``pilt_obj.epochs_E``
    * ``model_cfg["local_steps"]``                     (read by FLWorker)
    * ``sim_cfg["worker_compute_ms"]``  +  per-worker ``compute_mean_ms``
      (rescaled proportionally when E changes, so the BSP simulator's
      compute-mask stays consistent with the new local-step count)

Nothing on the GA random seed, channel, dataset split or model init
is touched — the calibration is observational only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pilt_protocol import (
    PILTProtocol,
    GAConfig,
    SchedulingInstance,
    DualPopulationGA,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Public configuration / report dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoTuneConfig:
    """Public knobs for the built-in adjudicator.

    Defaults are tuned for the K=10, L=10, ResNet-50 reference setup but
    apply unchanged to ResNet-20 / VGG-16 — the calibrator measures the
    actual hardware costs, the only things that depend on the model are
    the absolute ms numbers.

    Attributes
    ----------
    enabled
        Master switch.  When False, ``calibrate`` is a no-op and
        returns a report with ``enabled=False``.  This is the
        "user opts out" path.
    target_overlap
        Fraction of T_cmp that GA may consume; we pick the largest
        (Np, G_max) such that  T_GA · safety_margin ≤ target_overlap · T_cmp.
        0.9 leaves a 10 % safety bubble for runtime jitter; 1.0 is
        aggressive (no safety bubble).
    safety_margin
        Multiplied into the observed T_GA to absorb GA-time variance
        across rounds (the GA inner loop has a stall-driven early-exit
        which makes per-round time non-trivial to predict from a single
        probe).
    n_compute_probes
        Number of timed SGD passes to run on the slowest worker
        (one warm-up pass is always added on top).  Median of the timed
        runs is used as T_cmp.
    n_ga_probes
        GA solver runs per (Np, G_max) candidate.  GA runtime has
        relatively low variance so the default of 1 is fine.
    min_local_steps / max_local_steps
        Bounds when the calibrator is forced to bump E because even the
        smallest GA grid point exceeds the T_cmp budget.  ``max`` caps
        how aggressive E-stretching can be — too large hurts FedAvg
        convergence.
    ga_grid_Np / ga_grid_Gmax
        Cross-product of these is the candidate space.  Probed in
        order of ascending (Np · G_max).
    verbose
        Print probe timings as they're collected.
    """
    enabled         : bool   = False

    target_overlap  : float  = 0.90
    safety_margin   : float  = 1.10

    n_compute_probes: int    = 2
    n_ga_probes     : int    = 1

    min_local_steps : int    = 1
    max_local_steps : int    = 16

    ga_grid_Np      : Tuple[int, ...] = (20, 30, 50, 80)
    ga_grid_Gmax    : Tuple[int, ...] = (40, 60, 100, 150)

    verbose         : bool   = True


@dataclass
class AutoTuneReport:
    """Outcome of a calibration pass.  Suitable for pretty-printing."""
    enabled       : bool
    t_cmp_ms      : float = 0.0
    t_ga_ms       : float = 0.0
    chosen_Np     : int   = 0
    chosen_Gmax   : int   = 0
    chosen_E      : int   = 0
    saturated     : bool  = False        # True ⇒ even smallest GA > budget
    probe_table   : List[Tuple[int, int, float]] = field(default_factory=list)
    notes         : List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.enabled:
            return "[autotune] disabled — protocol parameters left untouched"

        head = (
            "[autotune] built-in adjudicator (内置判决模块) report\n"
            f"  T_cmp (real SGD on slowest worker)  = {self.t_cmp_ms:>7.1f} ms\n"
            f"  T_GA  (chosen GA configuration)     = {self.t_ga_ms:>7.1f} ms\n"
            f"  → chosen GA  Np / G_max             = {self.chosen_Np} / {self.chosen_Gmax}\n"
            f"  → chosen     local_steps E          = {self.chosen_E}\n"
            f"  saturated  (T_GA > budget at all)   = {self.saturated}"
        )
        rows = []
        if self.probe_table:
            rows.append("  GA grid probes (Np, G_max → T_GA ms):")
            for Np, Gm, t_ms in self.probe_table:
                rows.append(f"    Np={Np:>3d}  G_max={Gm:>3d}   {t_ms:>6.1f} ms")
        for n in self.notes:
            rows.append(f"  · {n}")
        return head + ("\n" + "\n".join(rows) if rows else "")


# ─────────────────────────────────────────────────────────────────────────────
#  The adjudicator itself
# ─────────────────────────────────────────────────────────────────────────────

class OverlapAutoTuner:
    """Single-shot calibrator that aligns T_GA with T_cmp.

    Usage (typically right before the BSP main loop, after pilt_obj +
    workers have been built):

        report = OverlapAutoTuner(AutoTuneConfig(enabled=True)).calibrate(
            pilt_obj=pilt_obj,
            workers=workers,
            ps=ps,
            model_cfg=cfg.MODEL,
            sim_cfg=cfg.SIMULATION,
        )
        print(report)

    All side-effects are scoped to the four containers passed in.  When
    ``enabled=False`` the call is a no-op.
    """

    def __init__(self, cfg: Optional[AutoTuneConfig] = None):
        self.cfg = cfg or AutoTuneConfig()

    # ── Public entry point ──────────────────────────────────────────────────

    def calibrate(self, *,
                  pilt_obj : PILTProtocol,
                  workers  : Sequence,
                  ps,
                  model_cfg: dict,
                  sim_cfg  : dict,
                  ) -> AutoTuneReport:
        rep = AutoTuneReport(
            enabled     = self.cfg.enabled,
            chosen_Np   = pilt_obj.cfg.ga.Np,
            chosen_Gmax = pilt_obj.cfg.ga.G_max,
            chosen_E    = pilt_obj.epochs_E,
        )
        if not self.cfg.enabled:
            rep.notes.append("autotune disabled — call site selected opt-out path")
            return rep
        if not workers:
            rep.notes.append("no workers handed in — cannot probe T_cmp")
            return rep

        if self.cfg.verbose:
            print("[autotune] probing T_cmp / T_GA on the live instance …",
                  flush=True)

        rep.t_cmp_ms = self._probe_t_cmp(workers, ps)
        if self.cfg.verbose:
            print(f"[autotune]   T_cmp = {rep.t_cmp_ms:.1f} ms",
                  flush=True)

        probes = self._probe_ga_grid(pilt_obj)
        rep.probe_table = probes
        if self.cfg.verbose:
            for Np, Gm, t_ms in probes:
                print(f"[autotune]   GA(Np={Np:>3d}, G_max={Gm:>3d}) "
                      f"= {t_ms:6.1f} ms", flush=True)

        # ── Pick GA settings ────────────────────────────────────────────────
        target_ms = self.cfg.target_overlap * rep.t_cmp_ms
        chosen: Optional[Tuple[int, int, float]] = None
        for Np, Gm, t_ga in probes:
            if t_ga * self.cfg.safety_margin <= target_ms:
                chosen = (Np, Gm, t_ga)        # keep the largest that fits

        if chosen is not None:
            rep.chosen_Np, rep.chosen_Gmax, rep.t_ga_ms = chosen
            rep.chosen_E = pilt_obj.epochs_E
            self._apply_ga(pilt_obj, rep.chosen_Np, rep.chosen_Gmax)
            rep.notes.append(
                f"GA fit at  Np={rep.chosen_Np}  G_max={rep.chosen_Gmax}  "
                f"(T_GA·{self.cfg.safety_margin:g}={rep.t_ga_ms*self.cfg.safety_margin:.1f}"
                f" ≤ {target_ms:.1f} ms)"
            )
            return rep

        # ── No grid point fits — try bumping E to extend T_cmp ──────────────
        Np_min, Gm_min, t_ga_min = probes[0]
        rep.chosen_Np, rep.chosen_Gmax, rep.t_ga_ms = Np_min, Gm_min, t_ga_min
        E_old = int(pilt_obj.epochs_E)
        E_new = self._needed_E_for_overlap(rep.t_cmp_ms, t_ga_min, E_old)

        if E_new > E_old:
            self._apply_ga(pilt_obj, Np_min, Gm_min)
            self._apply_E (pilt_obj, workers, model_cfg, sim_cfg, E_old, E_new)
            rep.chosen_E = E_new
            rep.notes.append(
                f"GA cost > T_cmp budget at every grid point; bumped "
                f"local_steps {E_old}→{E_new} (compute mask scaled "
                f"×{E_new/max(1,E_old):.2f}) so the smallest GA fits."
            )
        else:
            rep.chosen_E = E_old
            rep.saturated = True
            self._apply_ga(pilt_obj, Np_min, Gm_min)
            rep.notes.append(
                f"GA cost {t_ga_min:.1f} ms exceeds budget {target_ms:.1f} ms "
                f"at all grid points and E is already at "
                f"max_local_steps={self.cfg.max_local_steps}; using "
                f"smallest GA (Np={Np_min}, G_max={Gm_min}). Overlap will "
                f"be partial — ps_exposed_ms > 0 expected."
            )
        return rep

    # ── Probes ──────────────────────────────────────────────────────────────

    def _probe_t_cmp(self, workers: Sequence, ps) -> float:
        """Real wall-time of one SGD pass on the worker with the largest
        local dataset (the BSP straggler in expectation).
        """
        pivot = max(workers, key=lambda w: len(w.X))
        global_state = ps.get_global_model().state_dict()

        # Untimed warm-up so allocator / cuDNN heuristics settle.
        pivot.compute_gradient(global_state)

        samples: List[float] = []
        for _ in range(max(1, self.cfg.n_compute_probes)):
            t0 = time.perf_counter()
            pivot.compute_gradient(global_state)
            samples.append((time.perf_counter() - t0) * 1000.0)
        # Median is more robust than mean against e.g. background GC.
        return float(np.median(samples))

    def _probe_ga_grid(self, pilt_obj: PILTProtocol
                       ) -> List[Tuple[int, int, float]]:
        """Time the dual-population GA on a representative scheduling
        instance (uniform ε = E_total) for every (Np, G_max) candidate.

        The probe is purely observational — the live ``pilt_obj`` state
        (importance EMA, ε scheduler, last-best-perm cache) is not
        touched.
        """
        K, L = pilt_obj.K, pilt_obj.L

        # Representative ε: uniform global budget, before any importance
        # data has accumulated.  We deliberately don't call
        # pilt_obj.compute_ratios() because that mutates the ratio
        # scheduler's ε state; the calibration must be side-effect-free
        # on the protocol's internal trackers.
        eps_uniform = np.full(L, pilt_obj.cfg.E_total, dtype=np.float64)
        D = pilt_obj.target_payload_bytes(eps_uniform)

        T_cmp = np.zeros((K, L), dtype=np.int64)
        for k in range(K):
            for l in range(L):
                T_cmp[k, l] = pilt_obj.channel.compute_T_cmp(
                    k, l, pilt_obj.layer_compute_fraction,
                    int(pilt_obj.n_samples[k]), pilt_obj.epochs_E,
                    per_sample_compute_ms=pilt_obj.per_sample_ms,
                )
        inst = SchedulingInstance(
            task_payload_bytes = D,
            bytes_per_slot     = pilt_obj.channel.bytes_per_slot_per_rb,
            T_cmp              = T_cmp,
            n_rbs              = pilt_obj.cfg.ch.n_rbs,
        )

        base = pilt_obj.cfg.ga
        candidates = sorted(
            ((Np, Gm) for Np in self.cfg.ga_grid_Np
                       for Gm in self.cfg.ga_grid_Gmax),
            key=lambda t: t[0] * t[1],
        )

        out: List[Tuple[int, int, float]] = []
        for Np, Gm in candidates:
            gc = GAConfig(
                Np            = Np,
                G_max         = Gm,
                pc_crossover  = base.pc_crossover,
                pm_mutation   = base.pm_mutation,
                migrate_every = base.migrate_every,
                stall_limit   = base.stall_limit,
                seed          = base.seed,
            )
            runs: List[float] = []
            for _ in range(max(1, self.cfg.n_ga_probes)):
                ga = DualPopulationGA(inst, gc)
                t0 = time.perf_counter()
                ga.solve()
                runs.append((time.perf_counter() - t0) * 1000.0)
            out.append((Np, Gm, float(np.median(runs))))
        return out

    # ── Mutators ────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_ga(pilt_obj: PILTProtocol, Np: int, Gmax: int):
        pilt_obj.cfg.ga.Np    = int(Np)
        pilt_obj.cfg.ga.G_max = int(Gmax)

    def _apply_E(self,
                 pilt_obj : PILTProtocol,
                 workers  : Sequence,
                 model_cfg: dict,
                 sim_cfg  : dict,
                 E_old    : int,
                 E_new    : int):
        """Bump local_steps everywhere a worker / scheduler reads it
        from, AND rescale the simulator's per-round compute mask so the
        BSP arithmetic stays internally consistent.
        """
        pilt_obj.epochs_E = int(E_new)
        model_cfg["local_steps"] = int(E_new)

        scale = float(E_new) / float(max(1, E_old))
        sim_cfg["worker_compute_ms"] = (
            float(sim_cfg.get("worker_compute_ms", 0.0)) * scale
        )
        for w in workers:
            if hasattr(w, "compute_mean_ms"):
                w.compute_mean_ms = float(w.compute_mean_ms) * scale

    def _needed_E_for_overlap(self, t_cmp_ms: float, t_ga_ms: float,
                              E_old: int) -> int:
        """Smallest integer E ≥ E_old that satisfies the overlap target.

        Assumes T_cmp scales linearly with E (true to first order for
        FedAvg local SGD on a fixed batch size).  Capped at
        ``max_local_steps``.
        """
        if t_cmp_ms <= 0:
            return E_old
        per_step = t_cmp_ms / max(1, E_old)
        need_total = t_ga_ms * self.cfg.safety_margin / max(1e-9, self.cfg.target_overlap)
        E_need = int(math.ceil(need_total / max(1e-9, per_step)))
        E_clamped = min(self.cfg.max_local_steps,
                        max(self.cfg.min_local_steps, E_need))
        return max(E_old, E_clamped)


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight self-test (run this module directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """A standalone self-test that exercises the GA-side probe path
    with synthetic workers — no PyTorch / NS3 required.

    The 'workers' here are duck-typed shims that mimic the bits of
    FLWorker the autotuner reads (X, compute_gradient, compute_mean_ms).
    """
    import numpy as _np

    # Build a minimal PILT instance — same shape as main.py uses.
    rng = _np.random.default_rng(0)
    K, L = 4, 6
    layer_sizes = [int(rng.integers(2_000, 30_000)) for _ in range(L)]
    pilt = PILTProtocol(
        n_workers              = K,
        n_layers               = L,
        layer_sizes            = layer_sizes,
        dataset_sizes_per_worker = [128] * K,
        epochs_E               = 1,
        per_sample_compute_ms  = 0.05,
    )
    # Shrink the GA so the self-test is fast.
    pilt.cfg.ga.Np = 20; pilt.cfg.ga.G_max = 30

    # Synthetic worker / PS shims
    class _DummyTensor:
        def __init__(self, n): self._n = n
        def __len__(self): return self._n

    class _DummyWorker:
        def __init__(self, n, mean_ms):
            self.X = _DummyTensor(n)
            self.compute_mean_ms = float(mean_ms)
        def compute_gradient(self, _state, **_kw):
            # Pretend an SGD pass takes ~5 ms (slept to be measurable).
            time.sleep(0.005)
            return [], 0, self.compute_mean_ms

    class _DummyPS:
        class _M:
            def state_dict(_self): return {}
        def get_global_model(self): return _DummyPS._M()

    workers = [_DummyWorker(256 + 32 * i, 50.0) for i in range(K)]
    ps      = _DummyPS()
    sim_cfg   = {"worker_compute_ms": 50.0, "straggler_std_ms": 5.0}
    model_cfg = {"local_steps": 1, "batch_size": 64, "lr": 0.02}

    print("=== OverlapAutoTuner self-test ==================================")
    tuner = OverlapAutoTuner(AutoTuneConfig(
        enabled         = True,
        target_overlap  = 0.9,
        n_compute_probes= 1,
        n_ga_probes     = 1,
        ga_grid_Np      = (10, 20, 40),
        ga_grid_Gmax    = (15, 30, 60),
        verbose         = True,
    ))
    rep = tuner.calibrate(pilt_obj=pilt, workers=workers, ps=ps,
                          model_cfg=model_cfg, sim_cfg=sim_cfg)
    print(rep)
    print("\nresulting pilt.cfg.ga ->",
          f"Np={pilt.cfg.ga.Np} G_max={pilt.cfg.ga.G_max} "
          f"E={pilt.epochs_E}")
    print("autotuner self-test OK.")
