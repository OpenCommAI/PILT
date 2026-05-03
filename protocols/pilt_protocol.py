"""protocols/pilt_protocol.py — PILT algorithm-side source.

Importance-aware top-|g| sparsification with error-feedback residual,
plus a dual-population GA scheduler over the time-frequency RB grid.
Only the algorithm layer lives here; network transport (packet loss,
RTT, incast) is delegated to the NS-3 backend.

Public interface used by main.py:

    PILTProtocol(n_workers, n_layers, layer_sizes, ..., cfg=PILTConfig)
        .compute_ratios()              -> ε_l per layer
        .target_payload_bytes(eps)     -> per-(worker, layer) byte budget
        .solve_schedule(D)             -> (start_slot table, makespan)
        .schedule_to_worker_plan(...)  -> per-worker start times
        .worker_encode(k, grads, eps)  -> (encoded_layers, masks, sent_bytes)
        .update_importance(avg_layers) -> updates v_l for next round

Configuration:

    PILTConfig(beta, d, E_total, eps_min, ga: GAConfig, ch: WirelessChannelCfg)
    GAConfig(Np, G_max, seed, ...)
    WirelessChannelCfg(bandwidth_per_rb_hz, n_rbs, slot_duration_ms, ...)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Importance tracker
# ─────────────────────────────────────────────────────────────────────────────

class PILTImportanceTracker:
    """L2-norm EMA with RMS normalization for per-layer importance.

        v_l ← β · v_l + (1-β) · ‖g_l‖₂ / √S_l
    """

    def __init__(self, n_layers: int, beta: float = 0.9):
        self.n_layers = n_layers
        self.beta = float(beta)
        self.v = np.zeros(n_layers, dtype=np.float64)
        self._initialized = False

    def update(self, grads_per_layer: Sequence[np.ndarray],
               layer_sizes: Sequence[int]) -> np.ndarray:
        """v_l ← β·v_l + (1-β)·‖g_l‖₂/√S_l."""
        assert len(grads_per_layer) == self.n_layers, \
            f"expected {self.n_layers} layers, got {len(grads_per_layer)}"
        new_norm = np.empty(self.n_layers, dtype=np.float64)
        for l, g in enumerate(grads_per_layer):
            g_flat = g.ravel() if hasattr(g, "ravel") else np.asarray(g).ravel()
            S_l = max(1, int(layer_sizes[l]))
            new_norm[l] = float(np.linalg.norm(g_flat)) / math.sqrt(S_l)

        if not self._initialized:
            self.v = new_norm.copy()
            self._initialized = True
        else:
            self.v = self.beta * self.v + (1.0 - self.beta) * new_norm
        return self.v.copy()

    def rank(self) -> np.ndarray:
        """
        Layer rank r_l ∈ {1,…,L}, higher v_l → rank 1 (most important).
        Ties broken by layer index (stable sort).
        """
        order = np.argsort(-self.v, kind="stable")  # descending
        r = np.empty(self.n_layers, dtype=np.int64)
        for rank_pos, idx in enumerate(order):
            r[idx] = rank_pos + 1
        return r


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Ratio scheduler
# ─────────────────────────────────────────────────────────────────────────────

class PILTRatioScheduler:
    """Rank-based linear compensation with common difference d.

    ε_l starts at E_total (uniform) and is updated each iteration as:

        c_l = d · ((L+1)/2 − r_l)
        ε_l ← clip(ε_l + c_l, ε_min, 1)
        rescale so  Σ ε_l · S_l ≤ E_total · Σ S_l
    """

    def __init__(self, n_layers: int, layer_sizes: Sequence[int],
                 d: float = 0.05, E_total: float = 0.5,
                 eps_min: float = 0.05):
        self.n_layers = n_layers
        self.layer_sizes = np.asarray(layer_sizes, dtype=np.float64)
        assert len(self.layer_sizes) == n_layers
        self.d = float(d)
        self.E_total = float(E_total)
        self.eps_min = float(eps_min)
        # initialise ε_l uniformly to the global budget
        self.eps = np.full(n_layers, self.E_total, dtype=np.float64)

    def step(self, rank: np.ndarray) -> np.ndarray:
        """Update ε_l using the current rank vector."""
        L = self.n_layers
        center = 0.5 * (L + 1)
        c = self.d * (center - rank.astype(np.float64))   # zero-sum
        self.eps = np.clip(self.eps + c, self.eps_min, 1.0)
        self._rescale_to_budget()
        return self.eps.copy()

    def _rescale_to_budget(self):
        """Enforce Σ ε_l·S_l ≤ E_total · Σ S_l by proportional down-scaling."""
        total_budget = self.E_total * self.layer_sizes.sum()
        current = float(np.dot(self.eps, self.layer_sizes))
        if current > total_budget and current > 0:
            scale = total_budget / current
            self.eps = np.clip(self.eps * scale, self.eps_min, 1.0)
            # floor may have pushed us back over; take one more pass
            current = float(np.dot(self.eps, self.layer_sizes))
            if current > total_budget and current > 0:
                self.eps *= total_budget / current


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Worker-side mask + historical retention + local accumulation
# ─────────────────────────────────────────────────────────────────────────────

class PILTWorkerResidual:
    """Per-worker residual buffer with top-|g| masking and error feedback.

    Per-layer state:
      * last_sent       — most recently transmitted tensor; reused as the
                          "historical value" for masked-out positions.
      * local_residual  — accumulated unsent variations; added to the
                          current gradient before masking so their
                          magnitude eventually crosses the top-ε threshold.
    """

    def __init__(self, layer_sizes: Sequence[int]):
        self.L = len(layer_sizes)
        self.layer_sizes = list(layer_sizes)
        self.last_sent: List[np.ndarray] = [
            np.zeros(s, dtype=np.float32) for s in layer_sizes
        ]
        self.local_residual: List[np.ndarray] = [
            np.zeros(s, dtype=np.float32) for s in layer_sizes
        ]

    def encode(self, grads: Sequence[np.ndarray],
               epsilon: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Encode per-layer gradients into (effective_l, mask_l, sent_elems_l).

        eff_flat[l]   — gradient to be aggregated at the PS (transmitted
                        positions take current value, masked positions take
                        the historical value).
        mask_flat[l]  — boolean mask, True for transmitted elements.
        sent_elems[l] — number of transmitted elements.
        """
        eff_flat: List[np.ndarray] = []
        mask_flat: List[np.ndarray] = []
        sent_elems: List[int] = []

        for l, g in enumerate(grads):
            g_flat = np.asarray(g).ravel().astype(np.float32, copy=False)
            S_l = int(g_flat.size)

            # An empty layer-group has no payload and no residual; emit
            # zero-sized outputs so downstream sizes stay consistent.
            if S_l == 0:
                eff_flat.append(g_flat)
                mask_flat.append(np.zeros(0, dtype=bool))
                sent_elems.append(0)
                continue

            g_total = g_flat + self.local_residual[l]

            eps_l = float(epsilon[l])
            k_l = max(1, int(round(eps_l * S_l)))
            k_l = min(k_l, S_l)

            if k_l >= S_l:
                mask = np.ones(S_l, dtype=bool)
            else:
                # argpartition is O(N), much faster than argsort for large tensors
                idx = np.argpartition(np.abs(g_total), S_l - k_l)[S_l - k_l:]
                mask = np.zeros(S_l, dtype=bool)
                mask[idx] = True

            g_prev = self.last_sent[l]
            eff = np.where(mask, g_total, g_prev)

            # Residual: zero out the sent positions of g_total in-place on a
            # cheap copy so the unsent variations are accumulated next round.
            new_residual = g_total.copy()
            new_residual[mask] = 0.0
            self.local_residual[l] = new_residual

            # Same algebra as `eff`; reuse to avoid a second np.where alloc.
            self.last_sent[l] = eff

            eff_flat.append(eff)
            mask_flat.append(mask)
            sent_elems.append(int(mask.sum()))

        return eff_flat, mask_flat, sent_elems


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Wireless channel model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WirelessChannelCfg:
    """Parameters for the outage-constrained Rayleigh channel."""
    bandwidth_per_rb_hz : float = 180e3   # one LTE-style RB = 180 kHz
    n_rbs               : int   = 50      # 50 RBs × 180 kHz = 9 MHz (~ 10 MHz)
    slot_duration_ms    : float = 1.0     # Δτ
    p_out               : float = 0.05    # outage probability
    tx_power_dbm        : float = 23.0    # 23 dBm per worker
    noise_psd_dbm_hz    : float = -174.0  # thermal floor
    path_loss_alpha     : float = 3.0     # urban macro
    distance_m_range    : Tuple[float, float] = (30.0, 150.0)


class PILTChannelModel:
    """Per-worker wireless model.

        T_cmp^{k,l}    — integer slot index at which layer l's gradient
                         is ready on worker k (causality bound).
        R_guar_{k,m}   — guaranteed bits per slot on RB m for worker k
                         under Rayleigh outage p_out.

    The per-RB bandwidth W is taken from cfg.bandwidth_per_rb_hz.
    All RBs share the same R_guar per worker (flat fading across the
    band; path-loss is the dominant factor).
    """

    def __init__(self, n_workers: int, cfg: WirelessChannelCfg,
                 rng: np.random.Generator):
        self.n_workers = n_workers
        self.cfg = cfg
        self.rng = rng

        # Draw per-worker distance → path loss → γ_th (constant per run)
        d_min, d_max = cfg.distance_m_range
        self.distances = rng.uniform(d_min, d_max, size=n_workers).astype(np.float64)

        # Pre-compute guaranteed bytes per slot per worker on a single RB
        self.bytes_per_slot_per_rb = self._compute_bytes_per_slot()

    def _compute_bytes_per_slot(self) -> np.ndarray:
        """Guaranteed rate × Δτ → bytes per slot, per worker.

        Single scalar per worker (flat fading across RBs).
        """
        W   = self.cfg.bandwidth_per_rb_hz
        N0_dbm_per_hz = self.cfg.noise_psd_dbm_hz
        P_tx_dbm      = self.cfg.tx_power_dbm
        alpha         = self.cfg.path_loss_alpha
        p_out         = self.cfg.p_out
        slot_s        = self.cfg.slot_duration_ms / 1000.0

        # Convert dBm → linear watts
        P_tx    = 10 ** ((P_tx_dbm - 30.0) / 10.0)
        N0      = 10 ** ((N0_dbm_per_hz - 30.0) / 10.0)  # W/Hz
        noise_W = N0 * W

        # Reference distance-1m path loss (free space at 1 m, 2 GHz ≈ -40 dB)
        # For simplicity use L(d) = d^alpha (in linear scale).
        path_loss = self.distances ** alpha

        rx_power = P_tx / path_loss                           # linear
        gamma_mean = rx_power / noise_W                       # expected SNR

        # Outage threshold:  γ_th = γ_mean · (-ln(1-p_out))
        gamma_th = gamma_mean * (-math.log(1.0 - p_out))

        # Rate on one RB (bits/sec)
        R_bps = W * np.log2(1.0 + gamma_th)
        # Bytes per slot on ONE RB
        bytes_per_slot = R_bps * slot_s / 8.0
        return bytes_per_slot.astype(np.float64)    # shape (K,)

    def compute_T_cmp(self, worker_id: int, layer_idx: int,
                      layer_compute_fraction: np.ndarray,
                      n_samples_k: int, epochs_E: int,
                      per_sample_compute_ms: float = 0.5) -> int:
        """Per-(worker, layer) compute completion time in slots.

            T_cmp^{k,l} = ⌈ E · n_k · f_l · τ_sample / Δτ ⌉

        where τ_sample is the per-sample forward+backward wall-clock and
        f_l is the cumulative fraction of compute done by layer l.
        """
        slot_ms = self.cfg.slot_duration_ms
        t_ms = (epochs_E * n_samples_k * layer_compute_fraction[layer_idx]
                * per_sample_compute_ms)
        return max(1, int(math.ceil(t_ms / slot_ms)))


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Dual-population GA scheduler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GAConfig:
    """Dual-population GA hyper-parameters."""
    Np           : int   = 50     # individuals per population
    G_max        : int   = 100    # generations
    pc_crossover : float = 0.85
    pm_mutation  : float = 0.25
    migrate_every: int   = 6      # bidirectional migration interval
    stall_limit  : int   = 4      # trigger migration on this many stagnant gens
    seed         : int   = 12345


@dataclass
class SchedulingInstance:
    """Problem data handed to the GA.

    task_payload_bytes  [K, L] — D̂_{k,l} target bytes for (worker, layer)
    bytes_per_slot      [K]    — guaranteed bytes per RB per slot, per worker
    T_cmp               [K, L] — causality lower bound in slots
    n_rbs               int    — M RBs in the frequency grid
    """
    task_payload_bytes: np.ndarray
    bytes_per_slot    : np.ndarray
    T_cmp             : np.ndarray
    n_rbs             : int

    @property
    def K(self) -> int:
        return self.task_payload_bytes.shape[0]

    @property
    def L(self) -> int:
        return self.task_payload_bytes.shape[1]

    def n_tasks(self) -> int:
        return self.K * self.L

    def slots_needed(self, k: int, l: int) -> int:
        """How many (t,m) cells task (k,l) requires at worker k's rate."""
        bps = max(1.0, float(self.bytes_per_slot[k]))
        payload = float(self.task_payload_bytes[k, l])
        return max(1, int(math.ceil(payload / bps)))


class DualPopulationGA:
    """Dual-population GA over time-frequency RB scheduling.

    Encoding:
        Individual = permutation of task IDs ∈ {0,…,K·L-1}.
        Decoding   = greedy left-justified parallel-machine scheduling:
                     for each task in order, distribute its `need` cells
                     across the M RBs starting at max(T_cmp, earliest
                     free slot per RB) using a min-heap.

    This encoding keeps all individuals feasible automatically and lets
    crossover / mutation operate on a simple integer vector, which is
    ~100× faster than the raw (t,m)-cell encoding and gives much tighter
    makespans for typical K·L scales (K=10, L=10).

    Fitness:
        F = 1 / (T_wireless + 1e-9)      (minimising makespan)
    Since T_comp and T_core are constants w.r.t. the chromosome, this is
    monotonically equivalent to minimising round wall-clock.
    """

    def __init__(self, inst: SchedulingInstance, cfg: GAConfig = GAConfig()):
        self.inst = inst
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Precompute task list and per-task slot count
        self.tasks: List[Tuple[int, int]] = [
            (k, l) for k in range(inst.K) for l in range(inst.L)
        ]
        self.slots_needed = np.array([
            inst.slots_needed(k, l) for k, l in self.tasks
        ], dtype=np.int64)
        self.t_min = np.array([
            inst.T_cmp[k, l] for k, l in self.tasks
        ], dtype=np.int64)
        self.n_tasks = len(self.tasks)

        # Pre-allocate M-sized scratch used inside `_makespan_only` so the
        # GA hot path is allocation-free.
        M = inst.n_rbs
        self._rb_end_scratch = np.zeros(M, dtype=np.int64)
        self._j_arr = np.arange(M, dtype=np.int64)

    # ── Decoding (priority permutation → (t,m) cells) ────────────────────────

    def _makespan_only(self, order: np.ndarray) -> int:
        """Vectorised parallel-machine makespan evaluator.

        For each task in permutation order: start-times on the M RBs are
        st[m] = max(t_min_k, rb_end[m]); fill `need` cells level-by-level.
        The finishing slot T for sorted st_sorted is the smallest integer
        satisfying  (j+1)·T − Σ_{i≤j} st_sorted[i] ≥ need  for some
        j ∈ [0, M-1]. Scanning j vectorially gives O(M) per task.
        """
        M = self.inst.n_rbs
        rb_end = self._rb_end_scratch
        rb_end.fill(0)
        j_arr = self._j_arr
        makespan = 0
        for task_id in order:
            t_min_k = int(self.t_min[task_id])
            need = int(self.slots_needed[task_id])

            st = np.maximum(t_min_k, rb_end)
            sort_idx = np.argsort(st, kind="stable")
            st_sorted = st[sort_idx]
            cumsum = np.cumsum(st_sorted)
            T_j = (need + cumsum + j_arr) // (j_arr + 1)
            lower_ok = T_j >= st_sorted
            if M > 1:
                upper_ok = np.empty(M, dtype=bool)
                upper_ok[:-1] = T_j[:-1] <= st_sorted[1:]
                upper_ok[-1] = True
            else:
                upper_ok = np.array([True])
            valid = np.where(lower_ok & upper_ok)[0]
            if valid.size == 0:
                T = int(st_sorted[-1] + need)
                active = M
            else:
                j_ch = int(valid[0])
                T = int(T_j[j_ch])
                active = j_ch + 1

            if T > makespan:
                makespan = T

            capacity = active * T - int(cumsum[active - 1])
            excess = capacity - need
            rb_end[sort_idx[:active]] = T
            if excess > 0:
                rb_end[sort_idx[active - excess: active]] = T - 1
        return makespan

    def _decode(self, order: np.ndarray
                ) -> Tuple[int, List[List[Tuple[int, int]]]]:
        """
        Exact (heap-based) decode that produces the per-task (t,m) cells.

        Used ONCE at the end of GA to materialise the best individual.
        """
        import heapq
        M = self.inst.n_rbs
        rb_end = np.zeros(M, dtype=np.int64)
        assignments: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_tasks)]
        makespan = 0

        for task_id in order:
            t_min_k = int(self.t_min[task_id])
            need = int(self.slots_needed[task_id])
            heap = [(max(t_min_k, int(rb_end[m])), m) for m in range(M)]
            heapq.heapify(heap)
            cells = assignments[task_id]
            for _ in range(need):
                slot, m = heapq.heappop(heap)
                cells.append((slot, m))
                if slot + 1 > makespan:
                    makespan = slot + 1
                heapq.heappush(heap, (slot + 1, m))
            for slot, m in heap:
                rb_end[m] = slot
        return makespan, assignments

    def fitness(self, order: np.ndarray) -> float:
        # Engineering note: the previous version maintained a permutation→
        # makespan dict cache.  Empirically, the cache hit rate was below
        # ~2 % because mutations and crossovers almost always produce
        # fresh permutations, so the dict-key bytes() conversion + lookup
        # was net-negative.  The makespan evaluator itself is already
        # O(n_tasks · M) and allocation-free thanks to scratch buffers.
        return 1.0 / (self._makespan_only(order) + 1e-9)

    # ── Seeding ──────────────────────────────────────────────────────────────

    def _elite_seed(self) -> np.ndarray:
        """Sort by T_cmp ascending, then load ascending (SPT)."""
        return np.lexsort((self.slots_needed, self.t_min))

    def _lpt_seed(self) -> np.ndarray:
        """Longest-Processing-Time first; ties broken by earliest T_cmp."""
        return np.lexsort((self.t_min, -self.slots_needed))

    def _diverse_seed(self) -> np.ndarray:
        order = np.arange(self.n_tasks)
        self.rng.shuffle(order)
        return order

    # ── Genetic operators on permutations ────────────────────────────────────

    def _ox_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order crossover (OX): preserves partial ordering of both parents."""
        n = self.n_tasks
        a, b = sorted(self.rng.integers(0, n, size=2).tolist())
        if a == b:
            b = min(n, a + 1)
        child = -np.ones(n, dtype=np.int64)
        child[a:b] = p1[a:b]
        used = set(child[a:b].tolist())
        pos = b % n
        for gene in np.concatenate([p2[b:], p2[:b]]):
            if gene in used:
                continue
            child[pos] = gene
            used.add(int(gene))
            pos = (pos + 1) % n
        return child

    def _segment_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Partial-mapped-style segment crossover for elite branch."""
        return self._ox_crossover(p1, p2)

    def _random_mask_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform crossover with repair for diversity branch."""
        n = self.n_tasks
        child = -np.ones(n, dtype=np.int64)
        mask = self.rng.random(n) < 0.5
        used = set()
        for i in range(n):
            gene = p1[i] if mask[i] else p2[i]
            if gene not in used:
                child[i] = gene
                used.add(int(gene))
        # Fill gaps with remaining tasks in p1 order
        remaining = [int(g) for g in p1 if int(g) not in used]
        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        return child

    def _left_shift_mutation(self, order: np.ndarray) -> np.ndarray:
        """Move a task to an earlier position to compact the schedule."""
        n = self.n_tasks
        src = int(self.rng.integers(1, n))
        dst = int(self.rng.integers(0, src)) if src > 0 else 0
        cand = order.copy()
        task = int(cand[src])
        cand = np.delete(cand, src)
        cand = np.insert(cand, dst, task)
        return cand

    def _random_mutation(self, order: np.ndarray) -> np.ndarray:
        """Swap two random positions."""
        cand = order.copy()
        i, j = self.rng.integers(0, self.n_tasks, size=2)
        cand[i], cand[j] = cand[j], cand[i]
        return cand

    def _tournament(self, pop: List[np.ndarray], fits: np.ndarray,
                    k: int = 3) -> np.ndarray:
        ids = self.rng.integers(0, len(pop), size=k)
        best = int(ids[0])
        for i in ids[1:]:
            if fits[int(i)] > fits[best]:
                best = int(i)
        return pop[best]

    def _roulette(self, pop: List[np.ndarray], fits: np.ndarray) -> np.ndarray:
        f = fits - fits.min() + 1e-9
        p = f / f.sum()
        idx = int(self.rng.choice(len(pop), p=p))
        return pop[idx]

    # ── Main solve loop ──────────────────────────────────────────────────────

    def solve(self, warm_start: Optional[np.ndarray] = None,
              extra_seed : Optional[np.ndarray] = None
              ) -> Tuple[List[List[Tuple[int, int]]], int, List[int]]:
        """Run dual-population GA.

        `warm_start`  — last round's best permutation, seeded as P_elite[1].
        `extra_seed`  — alternative heuristic (e.g. LPT), seeded as P_elite[2].
        """
        cfg = self.cfg

        elite_seed   = self._elite_seed()
        diverse_seed = self._diverse_seed()

        def _is_valid_perm(p):
            return (isinstance(p, np.ndarray) and p.size == self.n_tasks
                    and np.array_equal(np.sort(p), np.arange(self.n_tasks)))

        P_elite   = [elite_seed.copy() for _ in range(cfg.Np)]
        if _is_valid_perm(warm_start):
            P_elite[1] = warm_start.astype(np.int64, copy=True)
        if _is_valid_perm(extra_seed) and cfg.Np >= 3:
            P_elite[2] = extra_seed.astype(np.int64, copy=True)
        P_diverse = [diverse_seed.copy() for _ in range(cfg.Np)]
        for ind in P_diverse[1:]:
            for _ in range(3):
                i, j = self.rng.integers(0, self.n_tasks, size=2)
                ind[i], ind[j] = ind[j], ind[i]

        fits_E = np.array([self.fitness(ind) for ind in P_elite])
        fits_D = np.array([self.fitness(ind) for ind in P_diverse])

        best_global = elite_seed.copy()
        best_fit = self.fitness(best_global)
        history = [self._makespan_only(best_global)]
        stall = 0           # stall since last migration
        stall_global = 0    # stall since last global-best improvement

        for gen in range(cfg.G_max):
            # ── Elite branch (exploit) ─────────────────────────────────────
            new_E = [best_global.copy()]      # elitism
            new_fitE = [best_fit]
            while len(new_E) < cfg.Np:
                p1 = self._tournament(P_elite, fits_E, k=3)
                p2 = self._tournament(P_elite, fits_E, k=3)
                if self.rng.random() < cfg.pc_crossover:
                    child = self._segment_crossover(p1, p2)
                else:
                    child = p1.copy()
                if self.rng.random() < cfg.pm_mutation:
                    child = self._left_shift_mutation(child)
                new_E.append(child)
                new_fitE.append(self.fitness(child))
            P_elite = new_E
            fits_E = np.array(new_fitE)

            # ── Diversity branch (explore) ─────────────────────────────────
            new_D = [P_diverse[int(np.argmax(fits_D))].copy()]
            new_fitD = [fits_D.max()]
            while len(new_D) < cfg.Np:
                p1 = self._roulette(P_diverse, fits_D)
                p2 = self._roulette(P_diverse, fits_D)
                if self.rng.random() < cfg.pc_crossover:
                    child = self._random_mask_crossover(p1, p2)
                else:
                    child = p1.copy()
                if self.rng.random() < cfg.pm_mutation:
                    child = self._random_mutation(child)
                new_D.append(child)
                new_fitD.append(self.fitness(child))
            P_diverse = new_D
            fits_D = np.array(new_fitD)

            # ── Global best ────────────────────────────────────────────────
            cand_E_idx = int(np.argmax(fits_E))
            cand_D_idx = int(np.argmax(fits_D))
            cand_E = P_elite[cand_E_idx]; cand_E_fit = fits_E[cand_E_idx]
            cand_D = P_diverse[cand_D_idx]; cand_D_fit = fits_D[cand_D_idx]

            gen_best, gen_best_fit = (
                (cand_E, cand_E_fit) if cand_E_fit >= cand_D_fit
                else (cand_D, cand_D_fit))

            if gen_best_fit > best_fit + 1e-12:
                best_fit = gen_best_fit
                best_global = gen_best.copy()
                stall = 0
                stall_global = 0
            else:
                stall += 1
                stall_global += 1

            # ── Bidirectional migration ────────────────────────────────────
            if (gen + 1) % cfg.migrate_every == 0 or stall >= cfg.stall_limit:
                self._migrate(P_elite, fits_E, P_diverse, fits_D,
                              n=max(1, cfg.Np // 10))
                stall = 0

            history.append(self._makespan_only(best_global))

            # Early stopping: bound per-round solver overhead.
            if stall_global >= 3 * cfg.stall_limit and gen >= cfg.migrate_every:
                break

        mk, asgn = self._decode(best_global)
        # Expose the winning permutation for warm-starting the next call
        self.last_best_perm = best_global.copy()
        return asgn, mk, history

    def _migrate(self, P_elite, fits_E, P_diverse, fits_D, n: int):
        """Bidirectional migration between populations."""
        elite_order = np.argsort(-fits_E)         # best first
        divers_order = np.argsort(-fits_D)
        for i in range(n):
            # worst of elite  ← random from diverse
            wE = int(elite_order[-(i+1)])
            rD = int(self.rng.integers(len(P_diverse)))
            P_elite[wE] = P_diverse[rD].copy()
            fits_E[wE] = fits_D[rD]
            # random of diverse  ← best of elite
            rD2 = int(self.rng.integers(len(P_diverse)))
            bE = int(elite_order[i])
            P_diverse[rD2] = P_elite[bE].copy()
            fits_D[rD2] = fits_E[bE]


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Top-level PILT driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PILTConfig:
    """PILT configuration. Two solver-level hot starts are always enabled:

      * GA warm-start — re-use the previous round's best permutation.
      * LPT elite seed — Longest-Processing-Time-first ordering.

    Both seed only the GA's initial population.
    """
    # Importance / ratio
    beta     : float = 0.9
    d        : float = 0.05      # rank-update step
    E_total  : float = 0.5       # global transmission budget
    eps_min  : float = 0.05

    # GA
    ga : GAConfig = field(default_factory=GAConfig)

    # Channel
    ch : WirelessChannelCfg = field(default_factory=WirelessChannelCfg)


class PILTProtocol:
    """Top-level PILT controller used by main.py per BSP round."""

    def __init__(self, n_workers: int, n_layers: int,
                 layer_sizes: Sequence[int],
                 layer_compute_fraction: Optional[Sequence[float]] = None,
                 dataset_sizes_per_worker: Optional[Sequence[int]] = None,
                 epochs_E: int = 1,
                 per_sample_compute_ms: float = 0.5,
                 cfg: Optional[PILTConfig] = None):
        self.K = n_workers
        self.L = n_layers
        self.layer_sizes = np.asarray(layer_sizes, dtype=np.int64)
        assert len(self.layer_sizes) == n_layers
        self.cfg = cfg or PILTConfig()
        self.bytes_per_elem = 4   # fp32 gradient element
        self.epochs_E = epochs_E
        self.per_sample_ms = float(per_sample_compute_ms)

        # Default: linear cumulative compute fraction (rough proxy for
        # uniform-depth CNNs).  Real use should pass measured per-layer
        # forward+backward timings normalised to (0, 1].
        if layer_compute_fraction is None:
            layer_compute_fraction = np.linspace(
                1.0 / n_layers, 1.0, n_layers)
        self.layer_compute_fraction = np.asarray(
            layer_compute_fraction, dtype=np.float64)

        if dataset_sizes_per_worker is None:
            dataset_sizes_per_worker = [256] * n_workers
        self.n_samples = np.asarray(dataset_sizes_per_worker, dtype=np.int64)

        # Sub-systems
        self.importance = PILTImportanceTracker(n_layers, beta=self.cfg.beta)
        self.ratio_sched = PILTRatioScheduler(
            n_layers, self.layer_sizes,
            d=self.cfg.d, E_total=self.cfg.E_total, eps_min=self.cfg.eps_min,
        )
        self.residuals = [PILTWorkerResidual(self.layer_sizes.tolist())
                          for _ in range(n_workers)]
        self.channel = PILTChannelModel(
            n_workers, self.cfg.ch,
            rng=np.random.default_rng(self.cfg.ga.seed + 97),
        )

        # Diagnostics + warm-start cache
        self.last_schedule = None
        self.last_makespan = 0
        self._last_best_perm: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def update_importance(self, avg_grads_per_layer: Sequence[np.ndarray]) -> np.ndarray:
        """Feed the PS-aggregated (averaged) gradient tensors to update v_l."""
        return self.importance.update(avg_grads_per_layer,
                                      self.layer_sizes.tolist())

    def compute_ratios(self) -> np.ndarray:
        """Compute ε_l for the upcoming round."""
        r = self.importance.rank()
        return self.ratio_sched.step(r)

    def target_payload_bytes(self, eps: np.ndarray) -> np.ndarray:
        """D̂_{k,l} = ε_l · S_l · B  (same across workers since S_l is global)."""
        D = np.outer(np.ones(self.K),
                     eps * self.layer_sizes * self.bytes_per_elem)
        return D.astype(np.int64)

    def solve_schedule(self, D_kl: np.ndarray
                       ) -> Tuple[List[List[Tuple[int, int]]], int]:
        """Run the dual-population GA to get a pre-scheduling table."""
        T_cmp = np.zeros((self.K, self.L), dtype=np.int64)
        for k in range(self.K):
            for l in range(self.L):
                T_cmp[k, l] = self.channel.compute_T_cmp(
                    k, l, self.layer_compute_fraction,
                    int(self.n_samples[k]), self.epochs_E,
                    per_sample_compute_ms=self.per_sample_ms,
                )

        inst = SchedulingInstance(
            task_payload_bytes = D_kl,
            bytes_per_slot     = self.channel.bytes_per_slot_per_rb,
            T_cmp              = T_cmp,
            n_rbs              = self.cfg.ch.n_rbs,
        )
        ga = DualPopulationGA(inst, self.cfg.ga)
        # Both solver-level hot starts always on — see PILTConfig.
        warm = self._last_best_perm
        extra_seed = ga._lpt_seed()
        best, makespan, _ = ga.solve(warm_start=warm, extra_seed=extra_seed)

        self.last_schedule = best
        self.last_makespan = makespan
        self._last_best_perm = ga.last_best_perm
        return best, makespan

    def worker_encode(self, worker_id: int,
                      grads: Sequence[np.ndarray],
                      eps: np.ndarray
                      ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Mask the worker's per-layer gradients; return eff and mask lists."""
        return self.residuals[worker_id].encode(grads, eps)

    # ── Convenience schedule interpretation ──────────────────────────────────

    def schedule_to_worker_plan(self, schedule, eps: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the (task_id -> [(t,m), ...]) chromosome into:

          start_slot_per_worker[K]   — earliest slot used by any of the worker's
                                        L tasks  (stagger send-start)
          bytes_per_worker[K]        — sum of ε_l · S_l · B across layers
                                        (= D̂_{k,l} summed over l, same for all k)
        This is fed to the NS3 backend.
        """
        K = self.K
        start_slots = np.full(K, 10**9, dtype=np.int64)
        end_slots = np.zeros(K, dtype=np.int64)

        for task_id, cells in enumerate(schedule):
            k, _l = divmod(task_id, self.L)
            for (t, _m) in cells:
                if t < start_slots[k]:
                    start_slots[k] = int(t)
                if t > end_slots[k]:
                    end_slots[k] = int(t)
        start_slots[start_slots == 10**9] = 0

        bytes_per_worker = (eps * self.layer_sizes * self.bytes_per_elem).sum()
        bytes_per_worker = np.full(K, int(round(bytes_per_worker)),
                                   dtype=np.int64)
        return start_slots, bytes_per_worker


# ─────────────────────────────────────────────────────────────────────────────
#  7.  Lightweight self-test (run this module directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PILT self-test ==============================================")
    rng = np.random.default_rng(0)

    K, L = 10, 10
    layer_sizes = [rng.integers(5_000, 50_000) for _ in range(L)]
    pilt = PILTProtocol(
        n_workers = K,
        n_layers  = L,
        layer_sizes = layer_sizes,
        dataset_sizes_per_worker = [64] * K,       # one batch per local step
        epochs_E  = 1,                              # E = local steps
        per_sample_compute_ms = 0.02,               # RTX 4090, CIFAR batch
    )

    # Fake gradient: larger magnitude on "deeper" layers
    grads_all_workers = []
    for k in range(K):
        grads_k = [rng.normal(0.0, 0.1 + 0.05 * l, size=layer_sizes[l]).astype(np.float32)
                   for l in range(L)]
        grads_all_workers.append(grads_k)

    # 1. PS aggregates a fake "average" and updates importance
    avg_grads = [
        np.mean([grads_all_workers[k][l] for k in range(K)], axis=0)
        for l in range(L)
    ]
    v = pilt.update_importance(avg_grads)
    print(f"importance v:       {np.round(v, 4)}")
    print(f"rank:               {pilt.importance.rank()}")

    # 2. Compute ratios
    eps = pilt.compute_ratios()
    print(f"eps per layer:      {np.round(eps, 3)}")
    print(f"Σ ε·S / E_total·ΣS:  "
          f"{float((eps*np.asarray(layer_sizes)).sum() / (pilt.cfg.E_total*sum(layer_sizes))):.3f}")

    # 3. Solve schedule
    D = pilt.target_payload_bytes(eps)
    print(f"D̂_{{k,l}} (worker 0): {D[0]}")
    print(f"Running dual-population GA  (Np={pilt.cfg.ga.Np}, Gmax={pilt.cfg.ga.G_max}) ...")
    import time as _t
    t0 = _t.time()
    # Shrink GA for self-test speed
    pilt.cfg.ga.Np = 20; pilt.cfg.ga.G_max = 30
    sched, makespan = pilt.solve_schedule(D)
    ga_ms = (_t.time() - t0) * 1000
    print(f"GA done in {ga_ms:.1f} ms, makespan = {makespan} slots "
          f"({makespan * pilt.cfg.ch.slot_duration_ms:.1f} ms)")

    # 4. Worker encode
    eff, mask, sent = pilt.worker_encode(0, grads_all_workers[0], eps)
    print(f"worker 0 sent elements per layer: {sent}")
    print(f"worker 0 effective grad norms:    "
          f"{[round(float(np.linalg.norm(e)), 3) for e in eff]}")

    # 5. Check residual propagation: second round should re-send some dropped elements
    eff2, mask2, sent2 = pilt.worker_encode(0, grads_all_workers[0], eps)
    rebounds = sum(int(np.logical_and(mask2[l], ~mask[l]).sum()) for l in range(L))
    print(f"elements newly transmitted in round 2 (residual rebound): {rebounds}")

    start_slots, bytes_per_worker = pilt.schedule_to_worker_plan(sched, eps)
    print(f"start_slot per worker: {start_slots}")
    print(f"bytes per worker:      {bytes_per_worker[0]}")
    print("PILT self-test OK.")
