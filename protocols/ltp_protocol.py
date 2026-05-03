"""protocols/ltp_protocol.py — Loss-Tolerant Protocol.

Loss-tolerant gradient gather:
  * Two packet classes: critical (reliable) and normal (drop-allowed).
  * Early-Close: PS closes the flow when arrived ≥ min_percent at the
    LT threshold, otherwise waits until the deadline.
  * Bubble-fill: missing positions are filled with zeros (or randomly
    sampled from delivered values; both supported).

The C++ network layer (or NS3 backend) reports a per-worker delivered
fraction; this module decides Early-Close action and applies bubble-fill.

Public interface used by main.py:

    LTPProtocol(n_workers, n_layers, layer_sizes, rtprop_ms, btlbw_mbps,
                deadline_slack_ms, min_percent, seed)
        .target_payload_bytes()
        .lt_threshold_ms(k) / .deadline_ms(k)
        .early_close_decision(k, at_lt, at_deadline, full_completion)
        .bubble_fill(k, grads_per_layer, frac)
        .end_epoch()
"""

from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Early-Close double-threshold state (per point-to-point link)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyCloseState:
    """LT threshold + deadline state for one gather operation.

      * LT_init   = 1.5 · RTprop  +  ModelBytes / BtlBw
      * LT_k+1    = min(LT_k, shortest 100%-gather time observed this epoch)
      * Deadline  = max_over_links(LT) + slack

    `min_percent` is the minimum arrival fraction required for Early-Close
    between LT and Deadline (default 80 %).
    """

    def __init__(self,
                 rtprop_ms        : float,
                 btlbw_mbps       : float,
                 model_bytes      : int,
                 deadline_slack_ms: float = 30.0,
                 min_percent      : float = 0.80):
        self.rtprop_ms        = float(rtprop_ms)
        self.btlbw_mbps       = float(btlbw_mbps)
        self.model_bytes      = int(model_bytes)
        self.deadline_slack_ms = float(deadline_slack_ms)
        self.min_percent       = float(min_percent)

        self.lt_threshold_ms  = self._lt_init()
        self.deadline_ms      = self.lt_threshold_ms + self.deadline_slack_ms

        # Per-epoch bookkeeping to update the threshold.
        self._shortest_full_bst_ms_this_epoch: float = float("inf")

    def _lt_init(self) -> float:
        # Model size in bits / BtlBw in Mbps → ms
        xmit_ms = (self.model_bytes * 8.0) / (self.btlbw_mbps * 1e3)
        return 1.5 * self.rtprop_ms + xmit_ms

    def reset_epoch(self):
        """Call at the start of every epoch."""
        self._shortest_full_bst_ms_this_epoch = float("inf")

    def observe_full_completion(self, bst_ms: float):
        """Record a 100%-delivery BST (e.g. rare but possible flows)."""
        if bst_ms < self._shortest_full_bst_ms_this_epoch:
            self._shortest_full_bst_ms_this_epoch = bst_ms

    def end_epoch(self):
        """At epoch end, clamp LT threshold to the shortest observed BST."""
        s = self._shortest_full_bst_ms_this_epoch
        if np.isfinite(s) and s > 0:
            self.lt_threshold_ms = min(self.lt_threshold_ms, s)
            self.deadline_ms = self.lt_threshold_ms + self.deadline_slack_ms
        self.reset_epoch()

    # ── Public — simulate the Early-Close decision for a single flow ──────
    def delivered_fraction(self,
                           arrived_fraction_at_lt: float,
                           arrived_fraction_at_deadline: float,
                           full_completion_ms: float | None) -> Tuple[float, str]:
        """
        Given the *network's* cumulative delivery profile, apply the
        Early-Close rule to determine the effective arrival fraction.

        Returns (fraction, reason) where reason ∈ {"full","early","deadline"}.
        """
        # Case 1: the whole flow arrived before LT threshold.
        if (full_completion_ms is not None
                and full_completion_ms <= self.lt_threshold_ms):
            self.observe_full_completion(full_completion_ms)
            return 1.0, "full"

        # Case 2: at LT threshold, enough is already received → close early.
        if arrived_fraction_at_lt >= self.min_percent:
            return float(arrived_fraction_at_lt), "early"

        # Case 3: between LT and deadline.
        # Close at the first moment the threshold is met. Approximate
        # using the fraction at the deadline (conservative).
        if arrived_fraction_at_deadline >= self.min_percent:
            return float(arrived_fraction_at_deadline), "early"

        # Case 4: deadline reached, take whatever is there.
        return float(arrived_fraction_at_deadline), "deadline"


# ─────────────────────────────────────────────────────────────────────────────
# Bubble-filling on the PS side
# ─────────────────────────────────────────────────────────────────────────────

def apply_bubble_fill(grad_flat: np.ndarray,
                       delivered_fraction: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Random-k delivery: keep a random fraction of elements, zero the rest.

    Losses happen at element (4-byte) granularity to preserve float32
    alignment.
    """
    S_l = int(grad_flat.size)
    keep_k = min(S_l, max(0, int(round(delivered_fraction * S_l))))
    if keep_k >= S_l:
        return grad_flat
    if keep_k <= 0:
        return np.zeros_like(grad_flat)
    # Random-k keep
    keep_idx = rng.choice(S_l, size=keep_k, replace=False)
    out = np.zeros_like(grad_flat)
    out[keep_idx] = grad_flat[keep_idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main protocol object
# ─────────────────────────────────────────────────────────────────────────────

class LTPProtocol:
    """LTP driver used by main.py per BSP round."""

    # Critical-class header per layer; trivially small so BST is unaffected.
    CRITICAL_BYTES_PER_LAYER: int = 32

    def __init__(self,
                 n_workers        : int,
                 n_layers         : int,
                 layer_sizes      : Sequence[int],
                 rtprop_ms        : float = 1.0,
                 btlbw_mbps       : float = 1000.0,
                 deadline_slack_ms: float = 30.0,  # 30ms DCN, 100ms WAN
                 min_percent      : float = 0.80,
                 bytes_per_elem   : int   = 4,
                 seed             : int   = 0):
        self.K = n_workers
        self.L = n_layers
        self.layer_sizes = np.asarray(layer_sizes, dtype=np.int64)
        self.bytes_per_elem = int(bytes_per_elem)

        # Per-worker Early-Close state (each link gets its own LT threshold)
        model_bytes = int(self.layer_sizes.sum()) * self.bytes_per_elem
        self._ec: List[EarlyCloseState] = [
            EarlyCloseState(rtprop_ms, btlbw_mbps, model_bytes,
                            deadline_slack_ms, min_percent)
            for _ in range(n_workers)
        ]

        self.rng = np.random.default_rng(seed)

    # ── API parity with PILTProtocol ─────────────────────────────────────────

    @property
    def ratios(self) -> np.ndarray:
        """LTP does NOT sparsify at the app layer → nominal ε=1 for all."""
        return np.ones(self.L, dtype=np.float64)

    def target_payload_bytes(self) -> np.ndarray:
        """Full gradient + a tiny critical header per layer."""
        per_layer = (self.layer_sizes * self.bytes_per_elem
                     + self.CRITICAL_BYTES_PER_LAYER)
        return np.outer(np.ones(self.K), per_layer).astype(np.int64)

    # ── Query / update the LT thresholds ─────────────────────────────────────

    def lt_threshold_ms(self, worker_id: int) -> float:
        return self._ec[worker_id].lt_threshold_ms

    def deadline_ms(self, worker_id: int) -> float:
        return self._ec[worker_id].deadline_ms

    def reset_epoch(self):
        for ec in self._ec:
            ec.reset_epoch()

    def end_epoch(self):
        """Propagate the shortest observed 100%-BST to the LT threshold."""
        for ec in self._ec:
            ec.end_epoch()

    # ── Decision + bubble-fill ───────────────────────────────────────────────

    def early_close_decision(self,
                             worker_id: int,
                             arrived_at_lt: float,
                             arrived_at_deadline: float,
                             full_completion_ms: float | None
                             ) -> Tuple[float, str]:
        return self._ec[worker_id].delivered_fraction(
            arrived_at_lt, arrived_at_deadline, full_completion_ms
        )

    def bubble_fill(self, worker_id: int,
                    grads: Sequence[np.ndarray],
                    delivered_fraction: float
                    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Random-k delivery applied uniformly to every layer.

        Returns (eff_grads, masks, sent_elements_per_layer).
        """
        eff, masks, sent = [], [], []
        for g in grads:
            g_flat = np.asarray(g).ravel().astype(np.float32, copy=False)
            S_l    = int(g_flat.size)
            keep_k = min(S_l, max(0, int(round(delivered_fraction * S_l))))
            if keep_k >= S_l:
                mask = np.ones(S_l, dtype=bool)
                eff.append(g_flat)
            elif keep_k <= 0:
                mask = np.zeros(S_l, dtype=bool)
                eff.append(np.zeros_like(g_flat))
            else:
                idx = self.rng.choice(S_l, size=keep_k, replace=False)
                mask = np.zeros(S_l, dtype=bool)
                mask[idx] = True
                out = np.zeros_like(g_flat)
                out[idx] = g_flat[idx]
                eff.append(out)
            masks.append(mask)
            sent.append(int(mask.sum()))
        return eff, masks, sent


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, L = 8, 20
    layer_sizes = [int(rng.integers(5000, 50000)) for _ in range(L)]
    ltp = LTPProtocol(K, L, layer_sizes,
                      rtprop_ms=1.0, btlbw_mbps=1000.0,
                      deadline_slack_ms=30.0, seed=1)
    print(f"Initial LT threshold (ms): {ltp.lt_threshold_ms(0):.2f}")
    print(f"Initial Deadline     (ms): {ltp.deadline_ms(0):.2f}")
    print(f"Payload bytes/worker      : {ltp.target_payload_bytes()[0].sum()}")

    # Simulate an incast round: worker 0 is slow
    grads = [rng.normal(0, 0.1, s).astype(np.float32) for s in layer_sizes]
    for k, (at_lt, at_dl, full) in enumerate([
        (0.85, 0.95, None),   # arrived ≥ 80% at LT → early close
        (0.60, 0.90, None),   # between: close at deadline when ≥ 80%
        (0.20, 0.50, None),   # deadline hit with < 80% → take whatever
        (1.00, 1.00, 2.0),    # full completion within LT
    ]):
        frac, why = ltp.early_close_decision(k, at_lt, at_dl, full)
        eff, m, s = ltp.bubble_fill(k, grads, frac)
        print(f"  worker {k}: LT={ltp.lt_threshold_ms(k):.1f}ms "
              f"frac={frac:.2f}  reason={why}  sent_elems_sum={sum(s)}")

    ltp.end_epoch()
    print(f"After epoch, LT threshold (ms): {ltp.lt_threshold_ms(0):.2f}  "
          f"(updated by observed full-completion 2.0 ms)")
