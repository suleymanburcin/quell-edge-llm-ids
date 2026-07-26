# QUELL — Quantized Edge LLM vs. Classical ML for IoT Intrusion Detection

Reproducibility package for the paper:

> **QUELL: An On-Device Benchmark of Quantized Edge LLMs versus Classical Machine Learning for IoT Intrusion Detection.** *(Under review, Computers & Security.)*

QUELL is, to our knowledge, the first **on-device, multi-axis benchmark** of a quantized
LLM-based intrusion detection system (IDS) on real IoT edge hardware. Lightweight decoder-only
LLMs (GPT-2-medium, Qwen2.5-0.5B/1.5B) are fine-tuned by serializing network-flow records as text,
quantized to FP16/INT8/4-bit, deployed on an **NVIDIA Jetson Orin Nano**, and compared against
well-tuned **XGBoost** and **random-forest** baselines on three heterogeneous IoT datasets under a
leakage-controlled, session-aware protocol.

**Headline result.** Under a matched, multi-seed protocol a simple, well-tuned classical detector
outperforms the quantized LLM on **every** axis — including clean-traffic accuracy on the dataset
most favourable to the LLM — while costing one to three orders of magnitude less on the same device.

---

## Evaluation axes
1. Detection quality (macro-F1, per-class)
2. Inference latency / throughput
3. Energy per inference
4. Peak memory
5. Unknown-attack generalization (leave-one-attack-out)
6. Adversarial-evasion robustness (noise sweep + black-box evasion)

## Datasets
Three public IoT/IIoT datasets, each evaluated **independently** (no pooling):

| Dataset | Classes | Split strategy |
|---|---|---|
| Edge-IIoTset | 15 | per-class temporal |
| CICIoT2023 | 34 | stratified + exact dedup |
| N-BaIoT | 11 | session-grouped (`__srcfile__`) |

Raw datasets are **not** redistributed here (size and licensing). Download them from their original
sources and place them under `data/raw/`. `scripts/03_preprocess_split.ipynb` regenerates the
processed tables and the versioned split indices; `results/split_report.json` documents the exact
split sizes, class distributions, and the columns dropped as leakage-prone.

## Repository layout
```
results/     Raw experimental outputs (JSON) — the source of truth for every number in the paper
figures/     Figures 1-6 (PNG), editable diagram sources (.drawio), and the scripts + data that
             regenerate the plotted figures, plus an integrity checker (verify_figures.py)
scripts/     Numbered pipeline (00 -> 09b): preprocessing, baselines, LLM fine-tuning,
             quantization, on-device benchmarking, generalization, and adversarial evaluation
src/quell/   Shared helpers (deterministic seeding)
configs/     Global configuration (seeds, paths)
docs/        Edge (Jetson) setup log
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| Hardware probe | `scripts/00_probe_hardware.py` | — |
| Download / verify raw data | `scripts/01_download_verify.ipynb` | raw-data manifest |
| Inspect | `scripts/02_inspect_data.ipynb` | — |
| Preprocess + leakage-safe split | `scripts/03_preprocess_split.ipynb` | `results/split_report.json` |
| Classical baselines (full-train) | `scripts/04_baseline_ml.ipynb` | `results/baseline_report.json` |
| LLM fine-tuning (feature-to-text + (Q)LoRA) | `scripts/05_*` | `results/llm_report.json` |
| Multi-seed fair comparison (Edge-IIoTset) | `scripts/05_5_multiseed_edge.ipynb` | `results/multiseed_report.json` |
| Matched baselines (CICIoT2023, N-BaIoT) | `scripts/05_6_baseline_matched.ipynb` | `results/baseline_matched_report.json` |
| Quantization + accuracy retention | `scripts/06_quantize_save.ipynb` | `results/quant_report.json` |
| On-device cost (Jetson) | `scripts/07_jetson_bench.py` (+ llama.cpp) | `results/jetson_fp16.json`, `jetson_llama_power.json`, `jetson_rf_cpu.json` |
| Unknown-attack generalization | `scripts/08_generalization_unseen.ipynb`, `08b_generalization_nbaiot.ipynb`, `08c_generalization_nbaiot_rest.ipynb` | `results/generalization_report.json`, `generalization_report_nbaiot_rest.json` |
| Adversarial robustness | `scripts/09_adversarial.ipynb`, `09b_adversarial_nbaiot.ipynb` | `results/adversarial_report.json`, `adversarial_report_nbaiot.json` |

## Results at a glance

| Axis | Random Forest / XGBoost | Quantized LLM (Qwen2.5-1.5B) |
|---|---|---|
| Clean macro-F1, Edge-IIoTset (5-seed) | **0.971 / 0.987** | 0.919 ± 0.013 |
| Clean macro-F1, CICIoT2023 (matched) | **0.748 / 0.762** | 0.571 |
| Clean macro-F1, N-BaIoT (matched) | **0.817 / 0.816** | 0.688 |
| On-device latency / energy (Jetson, batch-1) | **~20.8 ms / ~140 mJ** | 197 ms / 1374 mJ (INT8) |
| Unknown-attack mean recall (Edge-IIoTset, 14 folds) | **1.000** | 0.929 |
| Unknown-attack mean recall (N-BaIoT, 8 folds) | **0.986** | 0.935 |
| Adversarial macro-F1 at eps=0.1 (Edge-IIoTset) | **0.90** | 0.04 |
| Black-box evasion (Edge-IIoTset / N-BaIoT) | **0% / 24%** | 56% / 44% |

INT8 quantization is near-lossless (delta macro-F1 = -0.0002) and is required only to fit the model on the device.

## Figures and their integrity
Figures 4-6 are regenerated from the raw results by standalone scripts:

```
python figures/make_fig4.py     # cost-accuracy trade-off
python figures/make_fig5.py     # unknown-attack generalization
python figures/make_fig6.py     # adversarial robustness (two datasets)
```

Each script reads a small `figures/QUELL_Fig{4,5,6}_data.json` whose values are verbatim extracts of
`results/*.json` (the source file for every number is recorded in each JSON's `provenance` field).
To confirm that no plotted value was hand-edited or fabricated:

```
python figures/verify_figures.py     # asserts every figure value == the raw results/*.json
```

Figures 1-3 (pipeline, feature-to-text encoding, edge testbed) are vector diagrams; the editable
`.drawio` sources are included alongside the exported PNGs.

## Reproducibility notes
- Leakage-safe splits (session-aware / per-class temporal); no random split.
- Primary metric macro-F1; all preprocessing fit on the training partition only.
- Fixed seed; split indices versioned to disk; each dataset evaluated separately.
- Head-to-head detectors trained under a matched, balanced regime; multi-seed variance reported.
- **Training hardware:** HPC node with 2x NVIDIA RTX A5500.
- **Edge hardware:** NVIDIA Jetson Orin Nano (8 GB), JetPack 6.2, MAXN_SUPER with `jetson_clocks`;
  board power via `tegrastats` (VDD_IN). Quantized inference uses llama.cpp built from source
  (`sm_87`); the standard `bitsandbytes`/`torchao` INT8/4-bit paths do not load on this Tegra +
  torch 2.5 stack (see `docs/jetson_setup_log.md`).

## Environment
```
pip install -r requirements.txt
```
The Jetson uses the JetPack-provided PyTorch wheel rather than the pip package.

## Citation
```bibtex
@article{quell,
  title  = {QUELL: An On-Device Benchmark of Quantized Edge LLMs versus
            Classical Machine Learning for IoT Intrusion Detection},
  author = {TO BE ADDED},
  year   = {TO BE ADDED},
  note   = {Under review, Computers & Security}
}
```

## License
Code is released under the MIT License (see `LICENSE`). The datasets remain under their original licenses.
