#!/usr/bin/env python3
# ================================================================
# QUELL Step 07 - Jetson Orin Nano edge cost measurement (FP16) - v5
# ----------------------------------------------------------------
# v5 change (two Jetson-specific pitfalls fixed):
#   1) TensorFlow noise: importing transformers also imported the stray
#      TensorFlow on the Jetson (cuDNN/cuFFT 'already registered', protobuf
#      'MessageFactory GetPrototype'). USE_TF=0 disables the TF import.
#   2) NVML assert: the device_map={"":0} path triggers accelerate, which
#      queries GPU memory through NVML. Tegra does not fully support NVML ->
#      'NVML_SUCCESS == r INTERNAL ASSERT FAILED CUDACachingAllocator'.
#      Fix: no device_map. Load the model plainly, then .to("cuda").
#
#   torchao/bitsandbytes are still NOT used (broken on Jetson). FP16 only.
#
# OUTPUT: /tmp/jetson_fp16.json
# Power: tegrastats VDD_IN (mW). Energy/inf = avg_W * (time_s/N) * 1000 (mJ)
# ================================================================
import os
os.environ["USE_TF"] = "0"          # keep transformers from importing TF
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, time, subprocess, signal, re, gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = [
    ("gpt2-medium",        15),   # 355M  - fits comfortably (reference)
    ("Qwen/Qwen2.5-0.5B",  15),   # 0.5B  - fits
    ("Qwen/Qwen2.5-1.5B",  15),   # 1.5B  - borderline
]
WARMUP = 5
ITERS  = 50
MAX_LEN = 256
SAMPLE_TEXT = ("Network traffic flow. proto=tcp, duration=0.42, sbytes=1240, "
               "dbytes=980, sttl=64, dload=1.2, spkts=14, dpkts=12 . Attack type:")

class PowerMeter:
    def __init__(self, interval_ms=100):
        self.interval = interval_ms; self.proc = None; self.logf = None
        self.path = "/tmp/_quell_tegrastats.log"
    def start(self):
        self.logf = open(self.path, "w")
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval)],
                stdout=self.logf, stderr=subprocess.STDOUT); return True
        except FileNotFoundError:
            print("  [warning] tegrastats not found", flush=True); return False
    def stop(self):
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try: self.proc.wait(timeout=5)
            except Exception: self.proc.kill()
        if self.logf: self.logf.close()
    def avg_power_w(self):
        vals = []
        try:
            for line in open(self.path):
                m = re.search(r"VDD_IN\s+(\d+)mW", line) or re.search(r"POM_5V_IN\s+(\d+)mW", line)
                if m: vals.append(int(m.group(1)) / 1000.0)
        except FileNotFoundError: pass
        return (sum(vals)/len(vals), len(vals)) if vals else (None, 0)

def gpu_free_gb():
    try:
        free, total = torch.cuda.mem_get_info(); return free/1e9, total/1e9
    except Exception: return None, None

def bench_one(model_id, num_labels):
    print(f"\n=== {model_id} (FP16) ===", flush=True)
    f0, t0 = gpu_free_gb()
    if f0 is not None: print(f"  GPU free: {f0:.2f}/{t0:.2f} GB", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try: torch.cuda.reset_peak_memory_stats()
    except Exception: pass
    try:
        # No device_map - load fp16 to CPU first, then move to GPU manually (avoids NVML)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels, torch_dtype=torch.float16)
        model = model.to("cuda")
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        return {"model": model_id, "status": "load_failed", "error": str(e)[:200]}
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    dev = next(model.parameters()).device
    enc = tok(SAMPLE_TEXT, truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt").to(dev)
    with torch.no_grad():
        for _ in range(WARMUP): _ = model(**enc).logits
    torch.cuda.synchronize()
    pm = PowerMeter(100); have_power = pm.start(); time.sleep(0.3)
    lat = []
    with torch.no_grad():
        for _ in range(ITERS):
            torch.cuda.synchronize(); s = time.time()
            _ = model(**enc).logits
            torch.cuda.synchronize()
            lat.append((time.time() - s) * 1000.0)
    total_s = sum(lat) / 1000.0
    time.sleep(0.3); pm.stop()
    lat.sort()
    avg_ms = sum(lat)/len(lat); p95_ms = lat[int(0.95*len(lat))-1]
    fps = ITERS/total_s; peak_mb = torch.cuda.max_memory_allocated()/1e6
    avg_w, nsamp = pm.avg_power_w() if have_power else (None, 0)
    energy_mj = (avg_w * (total_s/ITERS) * 1000.0) if avg_w else None
    res = {"model": model_id, "status": "ok", "dtype": "fp16",
           "num_labels": num_labels, "iters": ITERS, "max_len": MAX_LEN,
           "latency_ms_avg": round(avg_ms,2), "latency_ms_p95": round(p95_ms,2),
           "throughput_fps": round(fps,2), "peak_gpu_mem_mb": round(peak_mb,1),
           "avg_board_power_w": round(avg_w,2) if avg_w else None,
           "energy_per_inf_mj": round(energy_mj,1) if energy_mj else None,
           "power_samples": nsamp}
    print(f"  latency avg={res['latency_ms_avg']}ms p95={res['latency_ms_p95']}ms "
          f"| {res['throughput_fps']} fps | peak mem {res['peak_gpu_mem_mb']}MB "
          f"| power {res['avg_board_power_w']}W | energy {res['energy_per_inf_mj']}mJ", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return res

def main():
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"), flush=True)
    f0, t0 = gpu_free_gb()
    if f0 is not None: print(f"GPU free at start: {f0:.2f}/{t0:.2f} GB", flush=True)
    results = []
    for mid, nl in MODELS:
        try: results.append(bench_one(mid, nl))
        except Exception as e:
            print(f"  ERROR {mid}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            results.append({"model": mid, "status": "error", "error": str(e)[:200]})
        gc.collect(); torch.cuda.empty_cache()
    out = Path("/tmp/jetson_fp16.json")
    json.dump({"device": "orin_nano_super", "results": results},
              open(out, "w"), indent=2, ensure_ascii=False)
    print("\n===== SUMMARY (FP16, Jetson) =====")
    print(f"{'model':22s} {'lat(ms)':>8s} {'fps':>7s} {'mem(MB)':>8s} {'power(W)':>8s} {'energy(mJ)':>10s}")
    for r in results:
        if r.get("status") == "ok":
            print(f"{r['model']:22s} {r['latency_ms_avg']:>8.1f} {r['throughput_fps']:>7.1f} "
                  f"{r['peak_gpu_mem_mb']:>8.0f} {str(r['avg_board_power_w']):>8s} {str(r['energy_per_inf_mj']):>10s}")
        else:
            print(f"{r['model']:22s}  -> {r.get('status')}: {r.get('error','')[:55]}")
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
