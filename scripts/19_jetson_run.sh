#!/usr/bin/env bash
# QUELL Step 19 (Jetson Orin Nano Super) — all on-device measurements in one run.
#  1. deployed fine-tuned detector (GGUF Q8_0 / Q4_K_M + trained head): macro-F1 + agreement with HF-FP16 on the
#     stratified Edge-IIoTset subset, batch-1 latency at the real prompt length, board power, energy, peak RAM
#  2. LLM cost at the real prompt lengths of CICIoT2023 and N-BaIoT (cost depends on length, not on weights)
#  3. trained RF / XGBoost: on-device macro-F1, batch-1 latency, batched throughput, power, energy (sklearn, XGBoost
#     native, ONNX Runtime if installed)
# Before running (once per boot):   sudo nvpmodel -m 2 && sudo jetson_clocks      (MAXN_SUPER, clocks pinned)
# Usage:   cd ~/jetson_bundle && bash scripts/19_jetson_run.sh 2>&1 | tee results/jetson_run.log
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; S="$B/scripts"; E="$B/edge_llm"; mkdir -p "$B/results"
LCPP="${LCPP:-$HOME/llama.cpp}"; COMMIT="$(cat "$E/llama_cpp_commit.txt" 2>/dev/null || echo 4413a192410fad1f054e9216114087a3e01527a0)"
N_ACC="${N_ACC:-1000}"; CTX=2112
log() { echo "[$(date '+%F %T')] $*"; }

log "device: $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
{ nvpmodel -q 2>/dev/null; jetson_clocks --show 2>/dev/null | head -5; } > "$B/results/power_mode.txt" || true
grep -qi "MAXN" "$B/results/power_mode.txt" || log "WARNING: power mode is not MAXN_SUPER -> run: sudo nvpmodel -m 2 && sudo jetson_clocks"
python3 -m pip install --user -q numpy pandas scikit-learn xgboost joblib >/dev/null 2>&1 || true
python3 -m pip install --user -q onnxruntime >/dev/null 2>&1 || log "onnxruntime not available (optional)"

# ---- llama.cpp, same commit as the export, CUDA build for Orin (sm_87) ----
if [ ! -x "$LCPP/build/bin/llama-server" ]; then
  log "building llama.cpp $COMMIT (CUDA, ~20-30 min)"
  [ -d "$LCPP" ] || git clone https://github.com/ggml-org/llama.cpp "$LCPP"
  git -C "$LCPP" fetch -q origin "$COMMIT" 2>/dev/null; git -C "$LCPP" checkout -q "$COMMIT" || log "WARNING: could not check out $COMMIT"
  cmake -S "$LCPP" -B "$LCPP/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$LCPP/build" -j"$(nproc)" --target llama-server
fi
git -C "$LCPP" rev-parse HEAD > "$B/results/llama_cpp_commit_jetson.txt" 2>/dev/null || true
log "llama.cpp: $(cat "$B/results/llama_cpp_commit_jetson.txt" 2>/dev/null) (export: $COMMIT)"

# ---- 1 + 2: deployed LLM detector ----
# The single-pass embedding graph reserves output buffers for every token of the ubatch (n_ubatch x vocab x 4 B), so the
# server is sized per input set: ubatch = longest sequence + margin. On the 8 GB Orin Nano a 1.2 GB buffer (ubatch ~1900)
# cannot be allocated; if the single-pass server for a length set does not start, that set is measured as a causal prefill
# in 512-token chunks through /completion (cost only; recorded in the result file as endpoint=completion).
maxlen() { python3 -c "import json,sys; print(max(len(json.loads(l)['ids']) for l in open(sys.argv[1])))" "$1"; }
up() { awk -v n="$1" 'BEGIN{print int((n+16+63)/64)*64}'; }
start_srv() {  # $1 model $2 ubatch $3 mode(embed|complete) $4 log ; complete: context 2048, ubatch $2
  local extra="--embeddings --pooling last" ctx="$2"; [ "$3" = complete ] && { extra=""; ctx=2048; }
  "$LCPP/build/bin/llama-server" -m "$1" $extra -lm none -c "$ctx" -b "$ctx" -ub "$2" -ngl 99 --port 8088 > "$4" 2>&1 &
  SRV=$!
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8088/health >/dev/null && return 0
    kill -0 $SRV 2>/dev/null || return 1; sleep 2
  done; return 1
}
stop_srv() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 5; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }
for T in q8_0 q4_k_m; do
  M="$E/model-$T.gguf"; [ -f "$M" ] || { log "missing $M"; continue; }
  UB=$(up "$(maxlen "$E/subset.jsonl")"); [ "$UB" -lt 512 ] && UB=512
  log "=== $T: accuracy on the Edge-IIoTset subset (ubatch $UB, all layers on GPU) ==="
  if start_srv "$M" "$UB" embed "$B/results/server_${T}_acc.log"; then
    python3 "$S/14_gguf_eval.py" --export "$E" --tag "$T" --device jetson --limit "$N_ACC" --power || log "accuracy run $T failed"
  else log "server for accuracy $T did not start (see server_${T}_acc.log)"; fi
  stop_srv
  for L in "$B"/lengths/*_ids.jsonl; do
    n=$(basename "$L" _ids.jsonl); UB=$(up "$(maxlen "$L")"); [ "$UB" -lt 512 ] && UB=512
    log "=== $T: cost at $n lengths (max $(maxlen "$L") tokens, ubatch $UB) ==="
    # single pass only up to ubatch 1024: at 1728 the server starts but the 8 GB board swaps and stalls (observed)
    if [ "$UB" -le 1024 ] && start_srv "$M" "$UB" embed "$B/results/server_${T}_${n}.log"; then
      python3 "$S/14_gguf_eval.py" --export "$E" --tag "$T" --device jetson --ids-file "$L" --power || log "cost run $T $n failed"
    else
      stop_srv; log "single-pass ubatch $UB does not fit -> causal prefill in 512-token chunks for $n"
      if start_srv "$M" 512 complete "$B/results/server_${T}_${n}_chunked.log"; then
        python3 "$S/14_gguf_eval.py" --export "$E" --tag "$T" --device jetson --ids-file "$L" --power --endpoint completion \
          || log "chunked cost run $T $n failed"
      else log "chunked server $T $n did not start"; fi
    fi
    stop_srv
  done
done
cp "$E"/gguf_eval_*_jetson*.json "$B/results/" 2>/dev/null
ls -l "$E"/*.gguf > "$B/results/gguf_sizes.txt"

# ---- 3: trained classical detectors ----
log "=== classical detectors (CPU) ==="
python3 "$S/19_jetson_classic_cost.py" --bundle "$B" || log "classical cost run failed"

tar -czf "$HOME/quell_jetson_results.tgz" -C "$B" results
log "DONE -> $HOME/quell_jetson_results.tgz   (send this file)"
