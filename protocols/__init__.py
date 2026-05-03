"""protocols/__init__.py — PILT / LTP / PLOT algorithm-side modules.

DCTCP (the reliable baseline) is handled inline in main.py.

The optional `autotuner` sub-module ships an opt-in calibrator that
profiles T_cmp / T_GA at startup and trims local_steps + GA params so
the GA wall-time hides inside worker SGD (overlap target).
"""
from .pilt_protocol import (
    PILTProtocol, PILTConfig, GAConfig, WirelessChannelCfg,
    PILTImportanceTracker, PILTRatioScheduler,
)
from .ltp_protocol  import LTPProtocol
from .plot_protocol import PLOTProtocol
from .autotuner     import (
    OverlapAutoTuner, AutoTuneConfig, AutoTuneReport,
)

__all__ = [
    "PILTProtocol", "PILTConfig", "GAConfig", "WirelessChannelCfg",
    "PILTImportanceTracker", "PILTRatioScheduler",
    "LTPProtocol", "PLOTProtocol",
    "OverlapAutoTuner", "AutoTuneConfig", "AutoTuneReport",
]
