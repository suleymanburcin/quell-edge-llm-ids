# QUELL — Jetson (Orin Nano Super) setup & measurement log

All steps, commands, and the problems/solutions encountered on the Jetson side, for
reproducibility (Step 07 — edge deployment + measurement).

## 0. Hardware / environment (probe output)
- Device: **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super**
- JetPack **6.2** / L4T **R36.4.7**, Ubuntu 22.04.5 (aarch64)
- CUDA **12.6** (/usr/local/cuda-12.6)
- CPU 6 cores, RAM **7.4 GB (shared CPU+GPU)** — ~4 GB free
- Python 3.10.12
- Power monitoring: **tegrastats present** (/usr/bin/tegrastats); no jtop
- Internet: available

Probe command (reference):
```bash
cat /proc/device-tree/model; head -1 /etc/nv_tegra_release; uname -m
nproc; free -h; ls -d /usr/local/cuda*; which tegrastats
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

## 1. GPU-enabled PyTorch (critical — the installed torch was newer than the driver, CUDA would not start)
Problem: the pip `torch` was built for a newer CUDA than the Jetson's CUDA 12.6 driver
→ "CUDA initialization: driver too old" → `torch.cuda.is_available()=False`.

Fix — cuSPARSELt + NVIDIA's Jetson torch wheel (JetPack 6.1–6.2, cp310, aarch64):
```bash
cd /tmp
CUS="libcusparse_lt-linux-aarch64-0.7.1.0-archive"
curl --retry 3 -OLs "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUS}.tar.xz"
tar xf "${CUS}.tar.xz"
sudo cp -a "${CUS}/include/"* /usr/local/cuda/include/
sudo cp -a "${CUS}/lib/"*     /usr/local/cuda/lib64/
sudo ldconfig
pip3 install --no-cache-dir --force-reinstall \
 "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
```
Result: **torch 2.5.0a0 (nv24.08), cuda available: True, device: Orin** ✓

## 2. Library compatibility (two pitfalls)
- **transformers too new (5.14.1)** → `ImportError: cannot import name 'DTensor'`
  (expects torch 2.6+). Fix: pin a version compatible with torch 2.5:
  ```bash
  pip3 install --no-cache-dir "transformers==4.46.3"
  ```
- **torchvision (0.27.1) left over from the old torch, broken with torch 2.5** →
  `RuntimeError: operator torchvision::nms does not exist` (crashes on transformers import).
  Not needed for text classification → remove:
  ```bash
  pip3 uninstall -y torchvision
  ```
  (Once removed, transformers sees `is_torchvision_available()`=False and skips that import chain.)

Note: the training host (HPC) had newer transformers/torch and no such issues. On the Jetson the
combination torch 2.5 + transformers 4.46.3 + (no torchvision) is used.

## 3. Quantization libraries are BROKEN on the Jetson (finding — reported in the paper)
Doing INT8/4-bit the "easy way" (Python libraries) on the edge was not possible on the Orin
Nano + JetPack 6.2 / torch 2.5. Two independent failures:

- **bitsandbytes (4-bit / 8-bit):** the GPU kernel does not load →
  `named symbol not found ... ops.cu`. There is no official prebuilt kernel for the Jetson
  (sm_87, aarch64); QLoRA/BnB quantization does not run on-device.
- **torchao (0.17.0):** incompatible with torch 2.5 →
  `Skipping import of cpp extensions ... upgrade to torch >= 2.11.0`.
  Worse, **merely importing it** corrupts the torch op namespace
  (`'_OpNamespace' ... _c10d_functional`), after which even transformers' Qwen2 module fails to
  import → torchao must not be touched at all.

**Conclusion (thesis finding):** LLM quantization at the edge is not library-portable; INT8 needs
the vendor's compilation path (NVIDIA TensorRT) or a GGUF/llama.cpp build. This is concrete
evidence for our "edge-deployment reality" contribution — the gap between the lab (HPC, x86 + new
torch) and the edge (Jetson).

## 4. FP16 measurement — clean path (no torchao)
Script: `scripts/07_jetson_bench.py`. Never imports torchao; measures FP16 only. Loads the model
directly (`torch_dtype=float16`) and then `model.to("cuda")` (avoids the NVML path). Model ladder:
gpt2-medium (355M) → Qwen2.5-0.5B → Qwen2.5-1.5B (whichever fits in ~4.8 GB).
Measured: latency (avg/p95 ms), throughput (fps), peak GPU memory (MB), average board power
(W, tegrastats VDD_IN), energy per inference (mJ). Output: `/tmp/jetson_fp16.json` → copied to `results/`.

### 4a. Two Jetson-specific pitfalls on the FP16 path
- **Stray TensorFlow:** importing transformers also imported the system TF on the Jetson →
  `cuDNN/cuFFT/cuBLAS already registered`, `TF-TRT`, protobuf
  `'MessageFactory' object has no attribute 'GetPrototype'`. Not fatal but noisy.
  Fix: `USE_TF=0 USE_TORCH=1` (transformers never loads TF).
- **NVML assert (the real blocker):** `device_map={"":0}` makes accelerate query GPU memory
  through NVML; since Tegra does not fully support NVML, torch raised
  `NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:838` and aborted the load
  (not OOM — 4 GB was free, gpt2 FP16 is ~0.7 GB). Fix: **do not use device_map**; load the model
  plainly, then `model.to("cuda")` (never enters the NVML path).

Run on device:
```bash
pip3 uninstall -y torchao        # it broke transformers — remove it
python3 scripts/07_jetson_bench.py
```

## 5. Quantized run — llama.cpp / GGUF (instead of bitsandbytes/torchao)
After the Python quantization stacks failed (see §3), llama.cpp was built from source
(Orin sm_87, CUDA). Qwen-1.5B was run in INT8 (Q8_0) and 4-bit (Q4_K_M) on the GPU — FP16 did not fit.

Build (nvcc was not on PATH — pass it explicitly):
```bash
sudo apt-get install -y cmake build-essential git libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
export PATH=/usr/local/cuda/bin:$PATH; export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j$(nproc)
```
Benchmark (256-token prefill = single-pass classification proxy):
```bash
./build/bin/llama-bench -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q8_0 -p 256 -n 32 -ngl 99
```
Power/energy: a tegrastats wrapper (VDD_IN) → `/tmp/jetson_llama_power.json`.

**Memory pitfall:** 8 GB shared; the NvMap GPU allocation needs truly free pages (not `available`).
`NvMapMemAllocInternalTagged error 12` = ENOMEM. Fix — drop caches before the run:
```bash
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"   # frees ~1.5 GB -> ~5 GB
```

Final results at the fixed MAXN_SUPER operating point (`nvpmodel -m 2` + `jetson_clocks`):
INT8 **197 ms / 6.97 W / 1374 mJ** (1.76 GiB); 4-bit **211 ms / 7.61 W / 1607 mJ** (1.04 GiB).
INT8 is faster in the compute-bound prefill regime (the 4-bit dequantization cost dominates there).
The classical random forest profiled on the same device (CPU) is in `results/jetson_rf_cpu.json`.

## Known limitation
An exact-model INT8 confirmation via the vendor path (TensorRT) and an ONNX cross-check were not
performed; the llama.cpp figures are used as the on-device cost proxy. This is stated in the paper's
threats-to-validity.
