#!/usr/bin/env bash
# QUELL Step 14 — convert the fine-tuned backbone (exported by 13_quantize_export.py) to GGUF F16 / Q8_0 / Q4_K_M.
# Usage: bash 14_gguf_make.sh <export_dir> [llama.cpp_dir]
#   export_dir = models/v4/<dataset>_<model>_s<seed>   (contains backbone_hf/, head.npy, subset.jsonl)
# Works without nvcc (CPU build): conversion/quantization do not need a GPU; the on-device
# (Jetson) evaluation uses the Jetson's own CUDA build of llama.cpp.
set -euo pipefail
EXP="$1"; LCPP="${2:-$HOME/llama.cpp}"
[ -d "$LCPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LCPP"
if ! command -v cmake >/dev/null 2>&1; then
  python -m pip install -q cmake          # cmake binary from PyPI (no system install needed)
fi
if [ ! -x "$LCPP/build/bin/llama-quantize" ] || [ ! -x "$LCPP/build/bin/llama-server" ]; then
  CUDA_FLAG=$(command -v nvcc >/dev/null 2>&1 && echo "-DGGML_CUDA=ON" || echo "")
  cmake -S "$LCPP" -B "$LCPP/build" $CUDA_FLAG -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$LCPP/build" -j"$(nproc)" --target llama-quantize llama-server
fi
# convert_hf_to_gguf.py uses the gguf-py package bundled in the llama.cpp checkout; we deliberately do
# NOT pip-install its requirements file, which pins torch/transformers and could alter the training env.
python -m pip install -q sentencepiece >/dev/null 2>&1 || true
[ -f "$EXP/model-f16.gguf" ]    || python "$LCPP/convert_hf_to_gguf.py" "$EXP/backbone_hf" --outtype f16 --outfile "$EXP/model-f16.gguf"
[ -f "$EXP/model-q8_0.gguf" ]   || "$LCPP/build/bin/llama-quantize" "$EXP/model-f16.gguf" "$EXP/model-q8_0.gguf"   Q8_0
[ -f "$EXP/model-q4_k_m.gguf" ] || "$LCPP/build/bin/llama-quantize" "$EXP/model-f16.gguf" "$EXP/model-q4_k_m.gguf" Q4_K_M
( cd "$LCPP" && git rev-parse HEAD ) > "$EXP/llama_cpp_commit.txt"
ls -lh "$EXP"/*.gguf
echo "llama.cpp commit: $(cat "$EXP/llama_cpp_commit.txt")"
