"""
federated/dataset.py — Dataset loaders and non-IID split utilities.

Provides:
  make_dataset()          — synthetic Gaussian-mixture classification
  make_cifar100_dataset() — CIFAR-100 from torchvision (downloads on first call)
  dirichlet_split()       — non-IID worker partition via Dirichlet(α)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def make_dataset(
    n_samples  : int,
    n_test     : int,
    n_features : int,
    n_classes  : int,
    noise      : float = 0.15,
    rng        : np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a harder classification dataset where gradient quality matters.

    Uses overlapping Gaussian clusters (noise=1.0 by default when called from
    config) so that:
      - Classes are NOT linearly separable → MLP needs many rounds to converge
      - Packet-loss that corrupts gradients visibly slows convergence
      - The three protocols produce meaningfully different accuracy curves

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    if rng is None:
        rng = np.random.RandomState(42)

    total = n_samples + n_test

    # Cluster centres — spacing 1.0 instead of 2.0 → heavy class overlap
    centres = rng.randn(n_classes, n_features) * 1.0

    X_list, y_list = [], []
    per_class = total // n_classes
    for c in range(n_classes):
        # noise=1.0 → within-class spread equals between-class spread → hard
        pts = rng.randn(per_class, n_features) * noise + centres[c]
        X_list.append(pts)
        y_list.append(np.full(per_class, c, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)

    # Shuffle
    idx   = rng.permutation(len(X))
    X, y  = X[idx], y[idx]

    # L2-normalise features
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X    /= norms

    return X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]


def dirichlet_split(
    y         : np.ndarray,
    n_workers : int,
    alpha     : float,
    rng       : np.random.RandomState,
) -> List[np.ndarray]:
    """
    Non-IID split using a Dirichlet(α) distribution over classes.

    Small α → each worker sees mostly one class (highly skewed).
    Large α → roughly IID.

    Returns list of index arrays, one per worker.
    """
    n_classes   = int(y.max()) + 1
    class_idxs  = [np.where(y == c)[0] for c in range(n_classes)]

    worker_idxs: List[List[int]] = [[] for _ in range(n_workers)]

    for cls_idx in class_idxs:
        rng.shuffle(cls_idx)
        # Dirichlet proportions for this class across workers
        proportions  = rng.dirichlet(alpha * np.ones(n_workers))
        splits       = (proportions * len(cls_idx)).astype(int)
        splits[-1]   = len(cls_idx) - splits[:-1].sum()   # fix rounding
        ptr = 0
        for w in range(n_workers):
            worker_idxs[w].extend(cls_idx[ptr : ptr + splits[w]].tolist())
            ptr += splits[w]

    return [np.array(idxs, dtype=np.int64) for idxs in worker_idxs]


def make_cifar100_dataset(
    data_root     : str   = "./data",
    rng           : np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-100 via torchvision (downloads on first call).

    Images are returned as float32 arrays of shape (N, 3072) — i.e. flattened
    and channel-normalised.  The model reshapes them back to (N, 3, 32, 32)
    internally (see ResNet20.forward).

    Returns
    -------
    X_train (50000, 3072), y_train (50000,),
    X_test  (10000, 3072), y_test  (10000,)
    """
    import os
    import torch
    from torch.utils.data import DataLoader

    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for CIFAR-100. "
            "Install with: pip install torchvision"
        ) from e

    os.makedirs(data_root, exist_ok=True)

    # CIFAR-100 channel statistics (per-channel mean and std)
    _mean = (0.5071, 0.4867, 0.4408)
    _std  = (0.2675, 0.2565, 0.2761)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    train_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=True,  download=True, transform=tfm)
    test_ds  = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=tfm)

    def _to_numpy(ds):
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy().reshape(len(x), -1).astype(np.float32))
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    X_train, y_train = _to_numpy(train_ds)
    X_test,  y_test  = _to_numpy(test_ds)
    return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)
