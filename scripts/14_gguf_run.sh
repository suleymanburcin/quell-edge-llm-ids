#!/usr/bin/env bash
# QUELL Step 14 (GPU host) — build GGUFs, then evaluate F16 / Q8_0 / Q4_K_M with the trained head
# and their agreement with the HF-FP16 classifier. Runs unattended; log to a file with nohup.
# Usage: bash 14_gguf_run.sh <export_dir> [limit]      (limit = number of subset rows, default all)
set -uo pipefail
EXP="$1"; LIMIT="${2:-0}"; LCPP="$HOME/llama.cpp"; HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/14_gguf_make.sh" "$EXP" "$LCPP" || { echo "GGUF build failed"; exit 1; }
CTX=$(python -c "import json;print(json.load(open('$EXP/meta.json'))['max_len']+64)")
NGL=$(command -v nvcc >/dev/null 2>&1 && echo 99 || echo 0)
for T in f16 q8_0 q4_k_m; do
  [ -f "$EXP/gguf_eval_${T}_ypb.json" ] && { echo "$T already done"; continue; }
  echo "=== $T (ctx $CTX, ngl $NGL) ==="
  "$LCPP/build/bin/llama-server" -m "$EXP/model-$T.gguf" --embeddings --pooling last \
      -c "$CTX" -b "$CTX" -ub "$CTX" -ngl "$NGL" -t "$(nproc)" --port 8088 > "$EXP/server_$T.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8088/health >/dev/null && break; sleep 2; done
  python "$HERE/14_gguf_eval.py" --export "$EXP" --tag "$T" --device ypb --limit "$LIMIT"
  kill $SRV; wait $SRV 2>/dev/null; sleep 2
done
echo "ALL DONE"
