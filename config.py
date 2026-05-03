"""config.py — Global configuration for the NS3-AI experiments.

Defaults:
    K            = 10 workers
    Model        = ResNet-50 (97 MB gradient) — VGG-16 (528 MB) via --model vgg16
    Dataset      = CIFAR-100
    Wireless     = 10 MHz total, Δτ = 1 ms
    Backbone     = 10 Gbps wired (10 µs base delay)
    d            = 0.05 rank-update step
    Np / Gmax    = 50 / 100  (dual-population GA)

SGD with Nesterov momentum, LR 0.02, weight-decay 1e-4, batch 64.
"""

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── CIFAR-100 Dataset ───────────────────────────────────────────────────────
DATASET = {
    "type"           : "cifar100",
    "n_features"     : 3072,        # 32×32×3 flattened
    "n_classes"      : 100,
    "n_workers"      : 10,          # K = 10
    "non_iid_alpha"  : 0.3,         # Dirichlet α (non-IID split)
    "data_root"      : "./data",    # torchvision download path
}

# ─── Model — ResNet-50 by default ────────────────────────────────────────────
MODEL = {
    "type"           : "resnet50",  # "resnet50" | "vgg16" | "resnet20"
    "n_classes"      : 100,
    # ResNet-50 / VGG-16 from scratch on CIFAR-100 (bilinear up-sampled
    # to 224×224) needs a gentler peak than the ResNet-20 recipe.
    "lr"             : 0.02,
    "momentum"       : 0.9,
    "weight_decay"   : 1e-4,
    "batch_size"     : 64,
    "local_steps"    : 10,
    "server_lr"      : 0.3,
    # Linear-warm-up applied to the base LR over the first `warmup_frac` of
    # wall-clock training time (standard ResNet-50 recipe to avoid early
    # divergence when the BN statistics are un-calibrated).
    "warmup_frac"    : 0.02,
}

# ─── Simulation wall-clock (the primary termination signal) ──────────────────
SIMULATION = {
    "num_rounds"           : 1_000_000,  # cap; time_limit is the real binding
    "n_workers"            : 10,
    "worker_compute_ms"    : 150.0,
    "straggler_std_ms"     : 20.0,
}

# ─── NS3-AI Network Channel ──────────────────────────────────────────────────
# Wireless uplink capacity per worker ~ 10 MHz × ~20 b/s/Hz ≈ 200 Mbps.
# Backbone = 10 Gbps wired pipe from switch to PS.
NETWORK = {
    "bandwidth_mbps"   : 200.0,
    "base_rtt_ms"      : 20.0,
    "packet_size_bytes": 1400,
    "buffer_size"      : 80,
}

# Stochastic AR(1) channel model — parameters live in fl-sim.cc.
CHANNEL_AR1 = {
    "loss_ar1_coef"    : 0.88,
    "loss_innovation"  : 0.40,
    "loss_min"         : 0.01,
    "loss_max"         : 0.40,
    "delay_ar1_coef"   : 0.85,
    "delay_innovation" : 70.0,
    "delay_min_ms"     : 2.0,
    "delay_max_ms"     : 60.0,
    "rng_seed"         : 42,
}
