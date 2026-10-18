# QUELL — Quantized Edge LLM vs. Classical ML for IoT Intrusion Detection

Reproducibility package (v4.0) for the paper:

> **QUELL: An On-Device Benchmark of Quantized Edge LLMs versus Classical Machine Learning for IoT Intrusion Detection.**
> S. B. Şüyun and F. Aktürk. *(Under review, IEEE Internet of Things Journal.)*

QUELL fine-tunes decoder-only LLMs (GPT-2-medium, Qwen2.5-0.5B/1.5B and Llama-3.2-1B) by serializing network-flow
records as text, quantizes them, deploys the fine-tuned detector on an **NVIDIA Jetson Orin Nano Super**, and compares it
with **random-forest** and **XGBoost** detectors on three IoT datasets (Edge-IIoTset, CICIoT2023, N-BaIoT) under a
leakage-controlled, multi-seed protocol with dataset-specific splits.

---

## Evaluation axes
1. Detection quality (macro-F1, per-class)
2. Inference latency / throughput
3. Energy per inference
4. Peak memory
5. Unknown-attack generalization (leave-one-attack-out and family hold-out)
6. Robustness (random-noise sensitivity and an attacker-constrained, format-controlled evaluation with HopSkipJump)

## Datasets
Three public IoT/IIoT datasets, each evaluated **independently** (no pooling):

| Dataset | Classes | Split strategy |
|---|---|---|
| Edge-IIoTset | 15 | per-class temporal |
| CICIoT2023 | 34 | stratified + exact dedup |
| N-BaIoT | 11 | session-grouped (`__srcfile__`) |

Raw datasets are **not** redistributed here (size and licensing). Download them from their original sources and place
them under `data/raw/`. `scripts/03_preprocess_split.ipynb` regenerates the processed tables and the versioned split
indices; `results/split_report.json` documents the exact split sizes, class distributions, and the columns dropped as
leakage-prone. On Edge-IIoTset the v4 results use the identifier-free feature set (45 features): the ten fields the
dataset authors drop before learning plus the Wireshark stream index `udp.stream`
(`scripts/quell_common.py`, `NOID_DROP`; `tcp.stream` is listed there too but is not a column of the dataset).

Files used for the reported results (downloaded by `scripts/01_download_data.ipynb`): Edge-IIoTset
`ML-EdgeIIoT-dataset.csv` (157,800 records); CICIoT2023 merged CSV release (Kaggle `akashdogra/ciciot23csv`, ≈46.7 M
records), of which `03_preprocess_split.ipynb` keeps every 31st record (1,506,018); N-BaIoT full version (Kaggle
`mkashifn/nbaiot-dataset`, 89 capture files), class-stratified random sample of 1,500,000 records (seed 42).
`03_preprocess_split.ipynb` reads the largest CSV under `data/raw/ciciot2023/`, i.e. the merged file; with only the
169 official part files present it would read a single part.
The split indices used in all runs are in `splits/`; `data/MANIFEST.md` lists the SHA-256 of every raw file and split file.

## Repository layout
```
results/      Raw experimental outputs (JSON) of the earlier versions, and results/tables/ (v3.0 tables)
results/v4/   Raw outputs of the v4 experiments — the source of truth for every number in the paper;
              PROVENANCE.md maps each file to its script, settings, log and manuscript table;
              tables/ holds Tables II-XI regenerated from the results
logs/v4/      Raw log of every v4 run (gzip); logs/v4/jetson/ for the on-device runs
scripts/      Numbered pipeline: 00 -> 09b (original), re_* (full-context re-runs, v3), 10 -> 23 (v4),
              plus shared helpers (quell_common.py, qlora_merge.py)
figures/      Figures 1-6 (PNG), editable diagram sources (.drawio), and make_fig{4,5,6}.py
src/quell/    Shared helpers (deterministic seeding)
configs/      Protocol summary (seeds, paths, splits, model settings)
splits/       Split indices (train/val/test) of the three datasets, as used in all runs
data/         MANIFEST.md: SHA-256 of the raw files and of the split indices (raw data not redistributed)
docs/         Edge (Jetson) setup log
reproduce.sh  One command: regenerate derived numbers, Tables II-XI and Figures 4-6; check every table value
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| Hardware probe | `scripts/00_probe_hardware.py` | — |
| Download raw data | `scripts/01_download_data.ipynb` (files used for the results); `scripts/01_download_verify.ipynb` (official sources, for checking) | raw-data manifest |
| Inspect | `scripts/02_inspect_data.ipynb` | — |
| Preprocess + leakage-safe split | `scripts/03_preprocess_split.ipynb` | `results/split_report.json` |
| Classical baselines (full-train) | `scripts/04_baseline_ml.ipynb` | `results/baseline_report.json` |
| LLM fine-tuning (feature-to-text + (Q)LoRA) | `scripts/05_*` | `results/llm_report.json` |
| Multi-seed comparison (Edge-IIoTset) | `scripts/05_5_multiseed_edge.ipynb` | `results/multiseed_report.json` |
| Matched baselines (CICIoT2023, N-BaIoT) | `scripts/05_6_baseline_matched.ipynb` | `results/baseline_matched_report.json` |
| Quantization + accuracy retention | `scripts/06_quantize_save.ipynb` | `results/quant_report.json` |
| On-device cost (FP16, Jetson) | `scripts/07_jetson_bench.py` | `results/jetson_fp16.json`, `jetson_llama_power.json`, `jetson_rf_cpu.json` |
| Unknown-attack generalization | `scripts/08_generalization_unseen.ipynb`, `08b_*`, `08c_*` | `results/generalization_report*.json` |
| Adversarial robustness | `scripts/09_adversarial.ipynb`, `09b_adversarial_nbaiot.ipynb` | `results/adversarial_report*.json` |
| Full-context re-runs (v3) | `scripts/re_05_4_detection.py`, `re_05_5_multiseed_edge.py`, `re_06_quantize.py`, `re_08_generalization.py`, `re_09_adversarial.py`, `re_09b_adversarial_nbaiot.py` | `results/*_fullctx.json` |
| Identifier-field control | `scripts/10_leakage_check.py` | `results/v4/leakage_check_edge_iiotset.json` |
| Token-length statistics | `scripts/11_token_stats.py` | `results/v4/token_stats.json` |
| Detection quality (seeds, Qwen2.5-0.5B, Llama-3.2-1B) | `scripts/12_detection.py` | `results/v4/llm_report_fullctx_*.json` |
| Fine-tuning, quantization retention, export | `scripts/13_quantize_export.py` (+ `qlora_merge.py`) | `results/v4/quant_retention_*.json` |
| GGUF conversion and evaluation | `scripts/14_gguf_make.sh`, `14_gguf_eval.py`, `14_gguf_run.sh` | `results/v4/gguf*/` |
| Unknown-attack generalization | `scripts/15_generalization.py` | `results/v4/generalization_*.json` |
| Random-noise sensitivity | `scripts/16_adversarial.py` | `results/v4/adversarial_*.json` |
| Attacker-constrained evaluation | `scripts/17_constrained_attack.py` | `results/v4/constrained_attack_*.json` |
| Jetson bundle | `scripts/18_jetson_bundle.py` | `results/v4/jetson_bundle_summary.json` |
| On-device measurements (Jetson) | `scripts/19_jetson_run.sh`, `19_jetson_classic_cost.py` | `results/v4/jetson/*.json` |
| On-device summary | `scripts/20_jetson_summary.py` | `results/v4/jetson/jetson_summary.json` |
| Derived statistics | `scripts/21_derived_stats.py` | `results/v4/derived_stats.json` |
| Table check | `scripts/22_verify_numbers.py` | — |
| Tables II–XI | `scripts/23_make_tables.py` | `results/v4/tables/*.csv` |

Exact environment variables and the log of each v4 run: `results/v4/PROVENANCE.md`.

## Results at a glance (v4)

| Axis | Random Forest / XGBoost | Quantized LLM |
|---|---|---|
| Clean macro-F1, Edge-IIoTset (identifier-free) | 0.884 / 0.895 | 0.892 ± 0.016 (Qwen2.5-1.5B), 0.901 ± 0.006 (Llama-3.2-1B) |
| Clean macro-F1, CICIoT2023 | 0.748 / 0.762 | **0.829 ± 0.005** |
| Clean macro-F1, N-BaIoT | **0.817 / 0.816** | 0.794 ± 0.026 |
| On-device latency / energy per flow (Edge-IIoTset, batch-1) | **0.07 ms / 0.52 mJ** (RF, ONNX Runtime), 2.53 ms / 20 mJ (XGBoost) | 278 ms / 4.70 J (INT8, llama.cpp) |
| Unknown-attack mean recall (CICIoT2023, 7 families) | 0.869 / 0.850 | 0.863 |
| Macro-F1 under random noise, ε = 0.1 (Edge-IIoTset) | **0.48 / 0.39** | 0.04 (0.21 format-controlled) |
| Constrained evasion, HopSkipJump ≤ 1σ | 0% / 0% | 0% |

Detection quality does not separate the LLMs from the classical detectors (on Edge-IIoTset the paired 95% CIs include
zero); the classical detectors are two to four orders of magnitude cheaper on the same device and less sensitive to
random input noise. On the device the 8-bit model reproduces the trained detector (99.1% agreement), while
post-training 4-bit quantization (Q4_K_M) loses 0.04 macro-F1.

## Reproducing the paper's tables and figures

Every reported number derives from the committed result files. No GPU, dataset download or training is needed:

```
pip install numpy scipy matplotlib
bash reproduce.sh
```

`reproduce.sh` recomputes the derived numbers (`results/v4/derived_stats.json`, `results/v4/jetson/jetson_summary.json`),
redraws Figures 4–6, writes Tables II–XI of the manuscript to `results/v4/tables/` (CSV, every cell computed from the
result files and formatted as printed), and runs `scripts/22_verify_numbers.py`, which checks the values printed in
Tables III–X against the result files (103 checks). A clean `git diff` afterwards means the regeneration is
bit-identical. Table I compares published studies and is not derived from results; Figures 1–3 are diagrams.

| RQ | Question | Table / Figure | Raw results from |
|---|---|---|---|
| RQ1 | Detection quality vs. classical ML | Table III | `10`, `12`, `21` |
| RQ2 | On-device cost (latency / energy / memory) | Tables V–VII, Fig. 4 | `07` (Table V), `13`, `14`, `18`, `19`, `20` |
| RQ3 | Unknown-attack generalization | Table VIII, Fig. 5 | `15` |
| RQ4 | Robustness and the effect of quantization | Tables IX–X, Fig. 6 | `16`, `17` |
| — | Accuracy retention under quantization | Table IV | `13` |

To regenerate the raw results from scratch, download the three datasets into `data/raw/`, run
`scripts/03_preprocess_split.ipynb`, then the numbered scripts with the settings listed in `results/v4/PROVENANCE.md`.
On the Jetson: build the bundle with `scripts/18_jetson_bundle.py` on the GPU host, copy it to the device and run
`bash scripts/19_jetson_run.sh` (MAXN_SUPER, `jetson_clocks`, desktop stopped); it builds llama.cpp at the export commit.

## Version history
- **v4.0** — results in `results/v4/`, scripts 10–23. Changes: identifier-free Edge-IIoTset features (the eleven
  session-identifying fields lifted the tree ensembles to 0.97–0.99 macro-F1); the fine-tuned detector itself
  (QLoRA adapter merged into the de-quantized base, converted to GGUF, run with its trained head) measured on the Jetson
  at the real prompt length of each dataset, instead of the base model with 256-token inputs; the trained classical
  detectors profiled on the same device (scikit-learn, ONNX Runtime, native XGBoost); additional seeds, Qwen2.5-0.5B and
  Llama-3.2-1B; XGBoost and CICIoT2023 added to the generalization, robustness and cost analyses; an
  attacker-constrained, format-controlled robustness evaluation with HopSkipJump at FP16, INT8 and NF4.
- **v3.0** — full-context re-runs (`scripts/re_*`, `results/*_fullctx.json`); v2 had truncated the serialized flows at
  256–512 tokens. `scripts/reproduce_tables.py` regenerates the v3.0 tables (`results/tables/`).
- **v2** — original pipeline (`scripts/00`–`09b`, `results/*.json` without suffix).

Earlier result files are kept unchanged for provenance.

## Reproducibility notes
- Dataset-specific splits: per-class temporal (Edge-IIoTset), class-stratified after exact deduplication (CICIoT2023),
  grouped by capture file (N-BaIoT).
- Primary metric macro-F1; label-dependent preprocessing is fit on the training partition only; categorical codes are a
  label-free mapping over all records.
- Fixed seeds; split indices versioned to disk; each dataset evaluated separately.
- Head-to-head detectors trained under a matched, balanced regime; ± values are sample standard deviations over seeds.
- **Training hardware:** HPC node with 2× NVIDIA RTX A5500.
- **Edge hardware:** NVIDIA Jetson Orin Nano Super (8 GB), JetPack 6.2, MAXN_SUPER with `jetson_clocks`; board power via
  `tegrastats` (VDD_IN). Quantized inference uses llama.cpp built from source (`sm_87`); the standard
  `bitsandbytes`/`torchao` INT8/4-bit paths do not load on this Tegra + torch 2.5 stack (see `docs/jetson_setup_log.md`).

## Environment
```
pip install -r requirements.txt
```
GPU host: Python 3.13, PyTorch 2.11 (CUDA 12.8), Transformers 5.14, PEFT, bitsandbytes, scikit-learn 1.9, XGBoost 3.3,
Adversarial Robustness Toolbox 1.20.1. Jetson: JetPack 6.2 (L4T R36.4.7), JetPack PyTorch wheel, llama.cpp commit 4413a19
(CUDA, sm_87), scikit-learn 1.7.2, XGBoost 3.2.0, ONNX Runtime 1.23.2.

## Citation
See `CITATION.cff`.

## License
Code is released under the MIT License (see `LICENSE`). The datasets remain under their original licenses.
