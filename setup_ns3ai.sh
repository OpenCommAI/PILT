#!/usr/bin/env bash
# setup_ns3ai.sh — Install dependencies and build NS3 with ns3-ai for FL simulation
# Run with:  sudo bash setup_ns3ai.sh
set -euo pipefail

echo "=== [1/4] Installing system dependencies ==="
apt-get install -y \
    libboost-all-dev \
    libsqlite3-dev \
    libgtk-3-dev \
    pkg-config \
    python3-dev

echo "=== [2/4] Installing Python packages ==="
pip3 install --break-system-packages pybind11 psutil torch numpy

echo "=== [3/4] Configuring NS3 with ns3-ai ==="
export PATH="$HOME/.local/bin:$PATH"
cd /home/goodlab/ns3

./ns3 configure \
    --build-profile=optimized \
    --enable-examples \
    --disable-tests \
    --enable-modules="core,network,internet,point-to-point,applications,flow-monitor,traffic-control,ai" \
    2>&1 | tail -20

echo "=== [4/4] Building ns3ai_fl_sim and Python binding ==="
./ns3 build ns3ai_fl_sim 2>&1 | tail -30

echo ""
echo "Build complete! Run the FL simulation with:"
echo "  cd /home/goodlab/FL && python3 main_ns3ai.py"
