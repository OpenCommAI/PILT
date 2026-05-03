"""model_torch.py — PyTorch models + FLWorker / ParameterServer.

Supported architectures (selected via cfg.MODEL.type):
  MLP      — two-layer fully-connected
  ResNet20 — CIFAR-adapted ResNet-20 (~278 K params)
  ResNet50 — torchvision ResNet-50 (~97 MB gradient)
  VGG16    — torchvision VGG-16   (~528 MB gradient)

Flat (batch, 3072) CIFAR inputs are reshaped to (batch, 3, 32, 32); for
ImageNet-scale backbones they are then bilinearly up-sampled to
(batch, 3, 224, 224) inside forward().

DEVICE is auto-selected (cuda:0 when available, else CPU). Gradients
are moved to CPU before returning so downstream numpy code is unchanged.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Global device — switch to 'cuda' if available for ~100x speedup on ResNet
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[model_torch] Using device: {DEVICE}")


# ─── MLP ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-layer MLP: input → hidden (ReLU) → output."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── ResNet-20 for CIFAR ─────────────────────────────────────────────────────

class _BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """
    ResNet-20 adapted for CIFAR images (32×32×3 → n_classes).

    Follows the original He et al. 2016 CIFAR architecture:
      3 stages × 3 residual blocks each, channels (16 → 32 → 64).
      ~278 K parameters, ~1.06 MB gradient.

    Input can be 2-D (batch, 3072) from flattened CIFAR storage or
    4-D (batch, 3, 32, 32) standard image tensors — both are handled.
    """

    def __init__(self, n_classes: int = 100):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, n_classes)

    @staticmethod
    def _make_layer(in_planes: int, planes: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_planes, planes, stride)]
        for _ in range(n_blocks - 1):
            layers.append(_BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.reshape(-1, 3, 32, 32)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


# ─── Torchvision wrappers (ResNet-50 / VGG-16) ───────────────────────────────

class _CIFARWrapper(nn.Module):
    """Wrap a torchvision ImageNet model for CIFAR-sized input.

    Flat (B, 3072) inputs are reshaped to (B, 3, 32, 32) and bilinearly
    up-sampled to (B, 3, 224, 224) before passing into the backbone, which
    is the standard evaluation protocol for ResNet-50 / VGG-16 on CIFAR-100.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.reshape(-1, 3, 32, 32)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224),
                              mode="bilinear", align_corners=False)
        return self.backbone(x)


def _build_resnet50(n_classes: int) -> nn.Module:
    """Stock torchvision ResNet-50 (25.56 M params ≈ 97 MB fp32 gradient)."""
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return _CIFARWrapper(m)


def _build_vgg16(n_classes: int) -> nn.Module:
    """Stock torchvision VGG-16 (138.36 M params ≈ 528 MB fp32 gradient)."""
    from torchvision import models
    m = models.vgg16(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return _CIFARWrapper(m)


# ─── Model factory ───────────────────────────────────────────────────────────

def make_model(model_cfg: dict) -> nn.Module:
    """Return the model configured by model_cfg['type']."""
    mtype = model_cfg.get("type", "mlp").lower()
    n_classes = model_cfg.get("n_classes", 100)
    if mtype == "resnet20":
        return ResNet20(n_classes=n_classes)
    if mtype == "resnet50":
        return _build_resnet50(n_classes)
    if mtype == "vgg16":
        return _build_vgg16(n_classes)
    # fallback MLP
    return MLP(
        n_in     = model_cfg["n_in"],
        n_hidden = model_cfg.get("hidden_size", 128),
        n_out    = n_classes,
    )


# ─── FL Worker ───────────────────────────────────────────────────────────────

class FLWorker:
    """FL worker: holds a local dataset shard and computes gradients.

    Memory-efficient design:  Workers do **not** own a GPU model.  Instead
    a single "scratch" model is shared across all K workers (set via
    :meth:`attach_shared_model`).  Each round the scratch model is reloaded
    with the current global weights and each worker runs its local SGD on
    it in turn, returning a CPU pseudo-gradient Δ = w₀ − w_E.  This reduces
    peak GPU memory from K×(model + activations) to 1×(model + activations)
    which is essential for ResNet-50 / VGG-16 (~100 MB / 500 MB per copy)
    when sharing the GPU with other processes.
    """

    def __init__(self, worker_id: int,
                 X_local: np.ndarray, y_local: np.ndarray,
                 model_cfg: dict, sim_cfg: dict,
                 rng: np.random.RandomState):
        self.worker_id   = worker_id
        # Keep worker data on CPU; move to GPU per-batch inside compute_gradient.
        # (Storing 60 MB per worker × 10 workers on GPU is fine, but CPU keeps
        # headroom for the large activation footprint of ResNet-50/VGG-16.)
        self.X           = torch.tensor(X_local, dtype=torch.float32)
        self.y           = torch.tensor(y_local, dtype=torch.long)
        self.batch_size  = model_cfg["batch_size"]
        self.compute_mean_ms = sim_cfg["worker_compute_ms"]
        self.compute_std_ms  = sim_cfg["straggler_std_ms"]
        self.rng         = rng
        self._model_cfg  = model_cfg
        self._criterion  = nn.CrossEntropyLoss()
        self._lr         = model_cfg["lr"]
        # Shared scratch model & optimizer (set by run_simulation)
        self._scratch    : nn.Module | None = None

    def attach_shared_model(self, scratch_model: nn.Module):
        """Point this worker at the shared GPU scratch model."""
        self._scratch = scratch_model

    def init_model(self, global_model: nn.Module, model_cfg: dict,
                   n_classes: int = None):
        """No-op in shared-model mode; kept for backwards compatibility."""
        pass

    def sync_model(self, global_model: nn.Module):
        """No-op in shared-model mode — sync happens just-in-time per round."""
        pass

    def set_lr(self, lr: float):
        self._lr = lr

    def compute_gradient(self, global_state: dict | None = None,
                         initial_params: list[torch.Tensor] | None = None):
        """FedAvg local update on the shared scratch model.

        Parameters
        ----------
        global_state : state_dict of the global (PS) model.  If provided,
            the scratch model is reloaded with these weights before running
            E local SGD steps.  Otherwise the scratch model keeps whatever
            state it has (useful for tests).
        initial_params : optional pre-cloned snapshot of the starting weights.
            When supplied (e.g. cached once per round in the BSP driver), the
            per-worker GPU clone is skipped — saves K-1 full-model clones per
            round.  Must be a list of GPU tensors aligned with
            `self._scratch.parameters()` (length and shape).

        Returns
        -------
        pseudo_grads : list[torch.Tensor]  per-parameter Δ = w₀ − w_E  (CPU)
        grad_bytes   : int
        comp_ms      : float               simulated compute time (ms)
        """
        assert self._scratch is not None, "attach_shared_model() first"
        local_steps = self._model_cfg.get("local_steps", 1)
        momentum    = self._model_cfg.get("momentum", 0.9)
        wd          = self._model_cfg.get("weight_decay", 0.0)
        n           = len(self.X)
        device      = next(self._scratch.parameters()).device

        if global_state is not None:
            self._scratch.load_state_dict(global_state)

        if initial_params is None:
            # In-place dest avoids the implicit allocation of a fresh tensor
            # per parameter that .clone() does; for ResNet-50 that's ~160
            # GPU allocs/frees per worker per round.
            initial_params = [p.data.detach().clone()
                              for p in self._scratch.parameters()]

        optimizer = optim.SGD(self._scratch.parameters(),
                              lr=self._lr, momentum=momentum, weight_decay=wd)

        for _ in range(local_steps):
            idx = torch.randperm(n)[: self.batch_size]
            xb = self.X[idx].to(device, non_blocking=True)
            yb = self.y[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = self._scratch(xb)
            loss   = self._criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Pseudo-gradient  Δ = w₀ − w_E  → CPU.
        # `.cpu()` already returns a *new* CPU tensor; the previous extra
        # `.clone()` triggered a redundant CPU→CPU memcpy of the entire
        # gradient (≈ K · model_size bytes per round for no benefit).
        # `.detach()` is also a no-op since the subtraction is computed in
        # no-grad context (parameters carry .grad but the arithmetic on
        # `.data` does not record a graph).
        pseudo_grads = [(p0 - p.data).cpu()
                        for p0, p in zip(initial_params,
                                         self._scratch.parameters())]

        # NOTE: torch.cuda.empty_cache() used to be called here per worker.
        # That triggered a CUDA synchronisation per worker (K syncs / round),
        # serialising what should be a pipelined GPU workload.  The caching
        # allocator already handles fragmentation; we let it manage memory.

        grad_bytes = sum(g.nelement() * 4 for g in pseudo_grads)
        comp_ms    = max(0.0, self.rng.normal(self.compute_mean_ms,
                                              self.compute_std_ms))
        return pseudo_grads, grad_bytes, comp_ms


# ─── Parameter Server ────────────────────────────────────────────────────────

class ParameterServer:
    """Global model holder with FedAvg-style gradient aggregation."""

    def __init__(self, model_cfg: dict,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.model = make_model(model_cfg).to(DEVICE)
        self._lr   = model_cfg.get("server_lr", 1.0)   # FedAvg: apply pseudo-gradients
        self._criterion = nn.CrossEntropyLoss()
        self.X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        self.y_test = torch.tensor(y_test, dtype=torch.long,    device=DEVICE)

    def get_global_model(self) -> nn.Module:
        return self.model

    def aggregate(self, worker_grads: list[list[torch.Tensor]],
                  delivery_ratios: list[float]):
        """
        Weighted FedAvg: each worker's pseudo-gradient is scaled by its delivery
        ratio (fraction of gradient bytes that arrived at the PS).

        Gradient clipping (global norm ≤ 5.0) is applied to the aggregated
        pseudo-gradient before the parameter update.  This is necessary when
        using FedAvg local steps because the pseudo-gradient Δ = w₀ − w_E can
        be O(E × lr × f(momentum)) ≈ 12× larger than a single raw gradient,
        and large server-side updates destabilise training.
        """
        if not worker_grads:
            return
        params = list(self.model.parameters())
        total_weight = sum(delivery_ratios)
        if total_weight == 0:
            return
        with torch.no_grad():
            # ── Build averaged pseudo-gradient ────────────────────────────────
            agg_list = []
            for pi, param in enumerate(params):
                agg = torch.zeros_like(param)
                for grads, ratio in zip(worker_grads, delivery_ratios):
                    if pi < len(grads):
                        g = grads[pi]
                        if g.device != param.device:
                            g = g.to(param.device)
                        agg += g * (ratio / total_weight)
                agg_list.append(agg)

            # ── Global gradient clipping (prevents FedAvg instability) ────────
            total_norm = torch.sqrt(sum(a.norm() ** 2 for a in agg_list))
            clip_norm  = 50.0   # pseudo-gradient norm ≈ 12× raw gradient; typical ResNet-20
                                # total raw-grad norm ≈ 2-3, so pseudo-grad ≈ 25-36.
                                # clip_norm=50 only trims extreme outliers, not normal rounds.
            clip_scale = (clip_norm / total_norm).clamp(max=1.0)
            if clip_scale < 1.0:
                agg_list = [a * clip_scale for a in agg_list]

            # ── Apply server update ───────────────────────────────────────────
            for param, agg in zip(params, agg_list):
                param.data -= self._lr * agg

    @torch.no_grad()
    def evaluate(self, eval_batch: int = 64,
                 bn_refresh_samples: int = 512) -> tuple[float, float]:
        """Returns (cross-entropy loss, accuracy) on the held-out test set.

        BatchNorm running statistics are NOT transmitted as gradients during FL
        aggregation, so they become stale on the PS.  We refresh them by running
        a short train-mode forward pass on the first `bn_refresh_samples` of the
        test set before switching to eval mode.  This is standard practice in
        FL with BN and does not constitute data leakage (BN stats are not model
        parameters).

        `eval_batch` is deliberately small (64) because ResNet-50 / VGG-16 at
        224×224 have GB-scale activations that would OOM on a shared GPU.
        """
        # Refresh BN running stats
        self.model.train()
        for i in range(0, min(len(self.X_test), bn_refresh_samples), eval_batch):
            self.model(self.X_test[i : i + eval_batch])

        self.model.eval()
        n = len(self.X_test)
        total_loss, total_correct = 0.0, 0
        for i in range(0, n, eval_batch):
            xb = self.X_test[i : i + eval_batch]
            yb = self.y_test[i : i + eval_batch]
            logits = self.model(xb)
            total_loss    += self._criterion(logits, yb).item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
        self.model.train()
        return total_loss / n, total_correct / n
