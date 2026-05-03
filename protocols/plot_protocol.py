"""protocols/plot_protocol.py — Fine-grained Packet Loss Tolerance.

Per-layer LTT-driven retransmission:
  * Send the full gradient over a packet-loss-tolerant channel.
  * PS measures per-layer delivered fraction.
  * Layers below their Layer-LTT trigger a retransmission round; layers
    above LTT keep their bubble-filled estimate.
  * Workers keep an RS-table indexed by layer for retx scheduling.

Public interface used by main.py:

    PLOTProtocol(n_workers, n_layers, layer_sizes, layer_ltt, seed)
        .target_payload_bytes()
        .simulate_delivery(grads_per_worker, delivered_frac_per_layer)
        .retx_layers_per_worker(delivered_frac_per_layer)
        .retx_payload_bytes()
        .apply_retx_delivery(grads, eff, mask, retx_layers, retx_frac)

Helpers:
    layer_ltt_resnet(...) / layer_ltt_vgg(...)
        Build a Layer-LTT vector from a model's block layout.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Layer-LTT derivation for ResNet / VGG from structural rules
# ─────────────────────────────────────────────────────────────────────────────

def layer_ltt_resnet(layer_sizes: Sequence[int],
                     block_boundaries: Sequence[Tuple[int, int]],
                     residual_layer_flags: Sequence[bool],
                     total_ltt: float = 0.20,
                     d_block_factor: float = 0.8,
                     rc_ratio: float = 0.5) -> np.ndarray:
    """Build per-layer LTT for a ResNet-family model.

    layer_sizes           : S_l per layer.
    block_boundaries      : list of (start, end) slices for conv blocks.
    residual_layer_flags  : True for residual-connected layers.
    total_ltt             : model-wide loss tolerance budget S(R).
    d_block_factor        : flatness control for the per-block sequence
                            (smaller → flatter, larger → steeper).
    rc_ratio              : LTT_residual / LTT_non_residual within a block.

    Returns L-vector with first and last layers = 0; blocks decreasing
    arithmetically; sum ≤ total_ltt.
    """
    L = len(layer_sizes)
    ltt = np.zeros(L, dtype=np.float64)

    k = len(block_boundaries)
    if k == 0:
        return ltt

    # (1) Block-level LTT — arithmetic sequence; first/last layers = 0.
    S_block = total_ltt
    # Upper bound on common difference d
    if k % 2 == 1:
        d_max = 2.0 * S_block / (k * (k - 1)) if k > 1 else 0.0
    else:
        d_max = 2.0 * S_block / (k * (k - 2)) if k > 2 else 0.0
    d = d_block_factor * d_max

    if k % 2 == 1:
        center = S_block / k
        first  = center + (k - 1) / 2 * d
    else:
        center = S_block / k                 # the two middle blocks
        first  = center + (k - 2) / 2 * d
    lr_block = np.array([first - i * d for i in range(k)], dtype=np.float64)
    lr_block = np.clip(lr_block, 0.0, None)
    # Renormalise to preserve budget exactly
    if lr_block.sum() > 0:
        lr_block *= S_block / lr_block.sum()

    # (2) Within-block LTT — RC layers tolerate less loss than non-RC.
    for b_idx, (s, e) in enumerate(block_boundaries):
        n_layers_in_block = e - s
        if n_layers_in_block <= 0:
            continue
        rc_mask = np.array([residual_layer_flags[i] for i in range(s, e)],
                           dtype=bool)
        x = int(rc_mask.sum())            # residual-connected count
        y = n_layers_in_block - x         # non-residual count

        # LRblock(i) = x · L_rc  +  y · L_ot   ,  L_rc = rc_ratio · L_ot
        # ⇒ L_ot = LRblock / (x·rc_ratio + y)
        denom = x * rc_ratio + y
        if denom <= 0:
            continue
        l_ot = lr_block[b_idx] / denom
        l_rc = rc_ratio * l_ot

        for j, idx in enumerate(range(s, e)):
            ltt[idx] = l_rc if rc_mask[j] else l_ot

    return ltt


def layer_ltt_vgg(layer_sizes: Sequence[int],
                  conv_idx: Sequence[int],
                  fc_idx  : Sequence[int],
                  total_ltt: float = 0.20,
                  d_conv_factor: float = 0.8,
                  d_fc_factor  : float = 0.5) -> np.ndarray:
    """Build per-layer LTT for VGG.

    10% of total budget goes to conv layers (decreasing across layer
    index); 90% goes to FC layers (increasing across layer index).
    """
    L = len(layer_sizes)
    ltt = np.zeros(L, dtype=np.float64)

    # conv: decreasing arithmetic
    n = len(conv_idx)
    if n > 0:
        S_conv = total_ltt * 0.10
        if n % 2 == 1:
            d_max = 2.0 * S_conv / (n * (n - 1)) if n > 1 else 0.0
        else:
            d_max = 2.0 * S_conv / (n * (n - 2)) if n > 2 else 0.0
        d = d_conv_factor * d_max
        center = S_conv / n
        if n % 2 == 1:
            first = center + (n - 1) / 2 * d
        else:
            first = center + (n - 2) / 2 * d
        vals = np.clip([first - i * d for i in range(n)], 0.0, None)
        if vals.sum() > 0:
            vals *= S_conv / vals.sum()
        for j, idx in enumerate(conv_idx):
            ltt[idx] = vals[j]

    # fc: INCREASING arithmetic
    m = len(fc_idx)
    if m > 0:
        S_fc = total_ltt * 0.90
        # Centre = S_fc/m; bounded common difference.
        if m == 1:
            ltt[fc_idx[0]] = S_fc
        else:
            d_fc = d_fc_factor * (S_fc / m)   # ≤ S_fc/m
            center = S_fc / m
            if m % 2 == 1:
                first = center - (m - 1) / 2 * d_fc
            else:
                first = center - (m - 2) / 2 * d_fc
            vals = np.clip([first + i * d_fc for i in range(m)], 0.0, None)
            if vals.sum() > 0:
                vals *= S_fc / vals.sum()
            for j, idx in enumerate(fc_idx):
                ltt[idx] = vals[j]

    return ltt


def default_resnet20_structure(n_layers: int) -> Tuple[List[Tuple[int, int]],
                                                       List[bool]]:
    """
    Produce (block_boundaries, residual_flags) for a stack of n_layers
    where the first and last layers are boundary (first-conv / fc) and
    the middle is split into 3 equal residual blocks (ResNet20 has 3
    stages each with 3 BasicBlocks ≈ 9 blocks; we keep it coarse here).

    The function is generic enough to work for layer-grouped projections
    of any size.  Residual flags mark every odd-indexed layer inside a
    block as RC (a common convention for BasicBlock where the 1st conv
    has the skip input and the 2nd is the post-add output).
    """
    if n_layers < 4:
        return [], [False] * n_layers
    # first layer (idx 0) and last layer (idx L-1) are non-block.
    block_start = 1
    block_end   = n_layers - 1
    n_block_layers = block_end - block_start
    # Split into 3 stages
    stage_sz = max(1, n_block_layers // 3)
    boundaries = []
    s = block_start
    for i in range(3):
        e = s + stage_sz if i < 2 else block_end
        if e > s:
            boundaries.append((s, e))
        s = e

    rc_flags = [False] * n_layers
    # Within each block, mark alternating layers as RC.
    for (s, e) in boundaries:
        for j in range(s, e):
            rc_flags[j] = ((j - s) % 2 == 0)   # first, third, ... = RC
    return boundaries, rc_flags


# ─────────────────────────────────────────────────────────────────────────────
# PLOT driver
# ─────────────────────────────────────────────────────────────────────────────

class PLOTProtocol:
    """PLOT driver used by main.py per BSP round.

    Per-round interaction with NS3:

    ┌──────────────┐  full gradients  ┌──────────────┐
    │  Workers     │ ───── AUDP ────► │      PS      │
    │ (RS-table)   │                  │ (loss count) │
    │              │ ◄── retx req ─── │              │
    │  retx layer  │ ───── AUDP ────► │              │
    └──────────────┘                  └──────────────┘
    """

    def __init__(self,
                 n_workers    : int,
                 n_layers     : int,
                 layer_sizes  : Sequence[int],
                 layer_ltt    : np.ndarray,
                 bytes_per_elem: int = 4,
                 seed         : int = 0):
        self.K = n_workers
        self.L = n_layers
        self.layer_sizes = np.asarray(layer_sizes, dtype=np.int64)
        self.bytes_per_elem = int(bytes_per_elem)
        self.layer_ltt = np.asarray(layer_ltt, dtype=np.float64)
        assert len(self.layer_ltt) == n_layers

        self.rng = np.random.default_rng(seed)

        # Per-worker RS-table: index-sets of lost positions per layer.
        # Stored as flat-index arrays (element-granularity) since the
        # simulated network operates at element granularity.
        self._rs_table: List[Dict[int, np.ndarray]] = [
            {} for _ in range(n_workers)
        ]

    # ── API parity ───────────────────────────────────────────────────────────

    @property
    def ratios(self) -> np.ndarray:
        """PLOT sends full gradients → ε=1 for all layers."""
        return np.ones(self.L, dtype=np.float64)

    def target_payload_bytes(self) -> np.ndarray:
        per_layer = self.layer_sizes * self.bytes_per_elem
        return np.outer(np.ones(self.K), per_layer).astype(np.int64)

    def retx_payload_bytes(self) -> Optional[np.ndarray]:
        """Bytes to re-send in the second pass, summed across workers.

        Returns `None` if no retransmission needed; else a [K] int array.
        """
        out = np.zeros(self.K, dtype=np.int64)
        any_retx = False
        for k, rs in enumerate(self._rs_table):
            if not rs:
                continue
            for l, idx in rs.items():
                out[k] += int(idx.size) * self.bytes_per_elem
            if out[k] > 0:
                any_retx = True
        return out if any_retx else None

    # ── Main per-round logic ─────────────────────────────────────────────────

    def simulate_delivery(self,
                          grads_per_worker: List[List[np.ndarray]],
                          delivered_per_layer: np.ndarray,
                          ) -> Tuple[List[List[np.ndarray]],
                                     List[List[np.ndarray]]]:
        """
        Given per-worker per-layer *delivered fraction* ∈ [0,1] from NS3,
        Random-k-drop unsent elements, populate RS-table, and return
        (eff_grads_per_worker, masks_per_worker).

        `delivered_per_layer`  shape = (K, L).
        """
        assert delivered_per_layer.shape == (self.K, self.L)
        eff_all: List[List[np.ndarray]] = []
        mask_all: List[List[np.ndarray]] = []
        for k in range(self.K):
            eff_k, mask_k = [], []
            self._rs_table[k].clear()
            for l, g in enumerate(grads_per_worker[k]):
                g_flat = np.asarray(g).ravel().astype(np.float32, copy=False)
                S_l = int(g_flat.size)
                frac = float(delivered_per_layer[k, l])
                frac = max(0.0, min(1.0, frac))
                keep_k = int(round(frac * S_l))
                if keep_k >= S_l:
                    mask = np.ones(S_l, dtype=bool)
                    eff = g_flat
                elif keep_k <= 0:
                    mask = np.zeros(S_l, dtype=bool)
                    eff = np.zeros_like(g_flat)
                    self._rs_table[k][l] = np.arange(S_l, dtype=np.int64)
                else:
                    idx_kept = self.rng.choice(S_l, size=keep_k, replace=False)
                    mask = np.zeros(S_l, dtype=bool)
                    mask[idx_kept] = True
                    eff = np.zeros_like(g_flat)
                    eff[idx_kept] = g_flat[idx_kept]
                    # The lost elements go into RS-table
                    lost_idx = np.setdiff1d(np.arange(S_l, dtype=np.int64),
                                            idx_kept, assume_unique=False)
                    self._rs_table[k][l] = lost_idx
                eff_k.append(eff)
                mask_k.append(mask)
            eff_all.append(eff_k)
            mask_all.append(mask_k)
        return eff_all, mask_all

    def retx_layers_per_worker(self,
                               delivered_per_layer: np.ndarray
                               ) -> List[List[int]]:
        """Layers needing retx per worker.

        A layer enters retx if its mean loss across workers exceeds its
        Layer-LTT and the worker actually has lost packets there.
        """
        # Per-layer mean loss across workers (broadcast recipient decision)
        loss_ratio_per_layer = 1.0 - delivered_per_layer.mean(axis=0)
        layers_to_retx = np.where(loss_ratio_per_layer > self.layer_ltt)[0].tolist()
        out: List[List[int]] = []
        for k in range(self.K):
            rs = self._rs_table[k]
            out.append([l for l in layers_to_retx
                        if l in rs and rs[l].size > 0])
        return out

    def apply_retx_delivery(self,
                            grads_per_worker: List[List[np.ndarray]],
                            eff_grads: List[List[np.ndarray]],
                            masks    : List[List[np.ndarray]],
                            retx_layers_per_worker: List[List[int]],
                            retx_delivered_fraction_per_layer: np.ndarray
                            ) -> Tuple[List[List[np.ndarray]],
                                       List[List[np.ndarray]]]:
        """Merge the second-pass (retx) delivery into eff_grads and masks.

        `retx_delivered_fraction_per_layer[k, l]` ∈ [0,1] is the fraction
        of the previously-lost set that was delivered this pass. Indices
        still lost remain in the RS-table.
        """
        for k in range(self.K):
            for l in retx_layers_per_worker[k]:
                lost_idx = self._rs_table[k].get(l)
                if lost_idx is None or lost_idx.size == 0:
                    continue
                n_lost = int(lost_idx.size)
                frac = float(retx_delivered_fraction_per_layer[k, l])
                frac = max(0.0, min(1.0, frac))
                n_recovered = int(round(frac * n_lost))
                if n_recovered <= 0:
                    continue
                recovered_idx = self.rng.choice(lost_idx, size=n_recovered,
                                                replace=False)
                g_flat = np.asarray(grads_per_worker[k][l]).ravel().astype(
                    np.float32, copy=False
                )
                eff_grads[k][l][recovered_idx] = g_flat[recovered_idx]
                masks[k][l][recovered_idx] = True
                # Update RS-table: remove recovered indices.
                remaining = np.setdiff1d(lost_idx, recovered_idx,
                                         assume_unique=False)
                self._rs_table[k][l] = remaining
        return eff_grads, masks


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    L = 20
    layer_sizes = [int(rng.integers(5000, 50000)) for _ in range(L)]

    bb, rc = default_resnet20_structure(L)
    print("block boundaries:", bb)
    print("rc flags        :", rc)
    ltt = layer_ltt_resnet(layer_sizes, bb, rc, total_ltt=0.20,
                           d_block_factor=0.8, rc_ratio=0.5)
    print(f"Layer-LTT (sum={ltt.sum():.3f}):")
    for i, v in enumerate(ltt):
        print(f"   layer {i:2d}  size={layer_sizes[i]:6d}  LTT={v:.4f}"
              + ("  [RC]" if rc[i] else ""))

    plot = PLOTProtocol(n_workers=4, n_layers=L,
                        layer_sizes=layer_sizes, layer_ltt=ltt, seed=2)
    print(f"payload bytes/worker: {plot.target_payload_bytes()[0].sum()}")

    # Simulate a network delivering varying fractions per worker and layer
    delivered = np.clip(rng.beta(5, 1, size=(4, L)), 0.2, 1.0)
    grads = [[rng.normal(0, 0.1, s).astype(np.float32)
              for s in layer_sizes] for _ in range(4)]
    eff, masks = plot.simulate_delivery(grads, delivered)
    retx = plot.retx_layers_per_worker(delivered)
    for k in range(4):
        print(f"worker {k}: retx layers = {retx[k]}  "
              f"retx bytes = {plot.retx_payload_bytes()[k]}")

    # Second pass: pretend NS3 delivered 90 % of the retx
    retx_frac = np.full((4, L), 0.9)
    eff2, masks2 = plot.apply_retx_delivery(grads, eff, masks, retx, retx_frac)
    after_retx = plot.retx_payload_bytes()
    print(f"After retx pass, still-lost bytes: "
          f"{after_retx.sum() if after_retx is not None else 0}")
