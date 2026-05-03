#!/usr/bin/env bash
# run_demo.sh -- one-shot reproducer.
#
#   1. Configure + build the C++ libraries (libpilt + DCTCP/LTP/PLOT
#      header-only) and unit tests.
#   2. Run unit tests (must all pass; otherwise abort).
#   3. Run the head-to-head 4-protocol benchmark and dump the JSON.
#   4. If results/metrics_*.npz exist (from main.py), pack them
#      together with the bench JSON into viz/dashboard_data.json.
#   5. Start a local HTTP server in cpp_proto/ at port 8080 and tell the
#      user to open http://localhost:8080/viz/dashboard.html.
#
# Designed so a fresh clone can do:
#
#     cd cpp_proto && bash run_demo.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RES_DIR="$ROOT/results"
BUILD="$HERE/build"
PY="${PYTHON:-/usr/bin/python3}"

echo "─── 1. cmake configure / build ────────────────────────────────────"
mkdir -p "$BUILD"
( cd "$BUILD" && cmake -DCMAKE_BUILD_TYPE=Release .. > cmake.log )
( cd "$BUILD" && make -j"$(nproc 2>/dev/null || echo 4)" )

echo
echo "─── 2. unit tests ─────────────────────────────────────────────────"
( cd "$BUILD" && ctest --output-on-failure )

echo
echo "─── 3. C++ 4-protocol benchmark ───────────────────────────────────"
"$BUILD/compare_protocols" \
    --K 10 --rounds 30 \
    --layers 1024,4096,16384,2048 \
    --loss_dctcp 0.0 --loss_ltp 0.45 --loss_plot 0.45 \
    --ltt 0.7  --pilt_e_total 0.5 \
    --out "$HERE/viz/proto_bench.json"

echo
echo "─── 4. pack sim + bench into dashboard_data.json ──────────────────"
if [ -d "$RES_DIR" ] && ls "$RES_DIR"/metrics_*.npz >/dev/null 2>&1; then
  echo "    found NS3-AI sim runs in $RES_DIR — including in dashboard"
  "$PY" "$HERE/viz/sim_to_json.py" \
      --sim_dir "$RES_DIR" \
      --bench   "$HERE/viz/proto_bench.json" \
      --out     "$HERE/viz/dashboard_data.json"
else
  echo "    no metrics_*.npz under $RES_DIR — bench-only dashboard"
  "$PY" "$HERE/viz/sim_to_json.py" \
      --sim_dir "$RES_DIR" \
      --bench   "$HERE/viz/proto_bench.json" \
      --out     "$HERE/viz/dashboard_data.json"
fi

echo
echo "─── 5. starting HTTP server ───────────────────────────────────────"
echo "    open  http://localhost:8080/viz/dashboard.html  in your browser"
echo "    (Ctrl-C to stop)"
exec "$PY" -m http.server --directory "$HERE" 8080
