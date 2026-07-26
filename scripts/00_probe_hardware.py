#!/usr/bin/env python3
"""
QUELL - Phase 0: hardware inventory.
Detects the capacity of the training machine (and the Jetson, if run there) and
prints a recommendation on which LLM sizes can be fine-tuned locally.

Usage:
    python3 scripts/00_probe_hardware.py

Has no dependencies (prints extra information if torch is installed).
"""
import json
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def bytes_to_gb(x):
    try:
        return round(int(x) / (1024 ** 3), 1)
    except Exception:
        return None


def get_ram_gb():
    # Linux
    meminfo = run("grep MemTotal /proc/meminfo")
    if meminfo:
        kb = int(meminfo.split()[1])
        return round(kb / (1024 ** 2), 1)
    # macOS
    mac = run("sysctl -n hw.memsize")
    if mac:
        return bytes_to_gb(mac)
    return None


def get_cpu():
    info = {
        "processor": platform.processor() or platform.machine(),
        "cores_logical": None,
    }
    try:
        import os
        info["cores_logical"] = os.cpu_count()
    except Exception:
        pass
    model = run("grep -m1 'model name' /proc/cpuinfo")
    if model:
        info["model"] = model.split(":", 1)[1].strip()
    return info


def get_gpus():
    gpus = []
    if shutil.which("nvidia-smi"):
        q = run("nvidia-smi --query-gpu=name,memory.total,driver_version "
                "--format=csv,noheader,nounits")
        if q:
            for line in q.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpus.append({
                        "name": parts[0],
                        "vram_gb": round(float(parts[1]) / 1024, 1),
                        "driver": parts[2] if len(parts) > 2 else None,
                    })
    return gpus


def get_torch():
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        return {"torch": "not installed"}


def is_jetson():
    model = run("cat /proc/device-tree/model 2>/dev/null")
    return bool(model and ("NVIDIA" in model and "Jetson" in model or "Orin" in model))


def recommend(gpus, ram_gb):
    """Which model size can be fine-tuned locally given the VRAM (assumes QLoRA 4-bit)."""
    if not gpus:
        return ("No GPU detected. Local LLM fine-tuning is hard. "
                "Classical ML baselines (XGBoost/RF) run here; "
                "for LLM training a cloud GPU (Colab/Kaggle/rented) is recommended.")
    vram = max(g["vram_gb"] for g in gpus)
    if vram >= 24:
        return f"~{vram}GB VRAM: 7B/8B models train comfortably with QLoRA. Full plan feasible."
    if vram >= 12:
        return f"~{vram}GB VRAM: 1B-3B QLoRA comfortable; 7B QLoRA borderline (small batch). Suggested focus: 1B + small transformer."
    if vram >= 8:
        return f"~{vram}GB VRAM: 1B QLoRA works; 7B+ hard. Start with GPT-2 / LLaMA-1B."
    return f"~{vram}GB VRAM: small models only (GPT-2 355M / TinyLlama). Consider cloud for 1B+."


def main():
    report = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu": get_cpu(),
        "ram_gb": get_ram_gb(),
        "gpus": get_gpus(),
        "torch": get_torch(),
        "is_jetson": is_jetson(),
    }
    report["recommendation"] = recommend(report["gpus"], report["ram_gb"])

    print("=" * 60)
    print("QUELL HARDWARE REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 60)
    print("RECOMMENDATION:", report["recommendation"])
    print("=" * 60)


if __name__ == "__main__":
    main()
