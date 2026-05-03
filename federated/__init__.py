"""federated/__init__.py

PyTorch-based FL building blocks used by main.py:

* dataset.py       — CIFAR-100 loading + Dirichlet non-IID split
* model_torch.py   — ResNet/VGG models, FLWorker, ParameterServer

Legacy NumPy MLP + async worker/PS live in ../legacy/federated/.
"""
from .dataset     import make_dataset, make_cifar100_dataset, dirichlet_split
from .model_torch import FLWorker, ParameterServer, make_model

__all__ = [
    "make_dataset", "make_cifar100_dataset", "dirichlet_split",
    "FLWorker", "ParameterServer", "make_model",
]
