# Provenance of the v4 results

For every file in `results/v4/`: the script that produced it, its settings, the raw log of the run (`logs/v4/`) and
where the result is used in the manuscript. All runs were on the GPU host (2× RTX A5500) unless noted. ± values are
**sample** standard deviations (ddof = 1). `queueN.log.gz` record the start and end time of each batch of runs.

Paths printed inside the logs are those of the working layout at run time: results in `results/rev1/`, scripts in
`scripts/rev1/r1_<letter>_*.py`. Result files were moved to `results/v4/` unchanged; scripts were renamed to step numbers (file paths, comments and
status messages updated; computations unchanged); letter → step:
D → 10, I → 11, C → 12, A → 13, B → 14, F → 15, E → 16, G → 17, H → 18, J → 19, L → 20, K → 21, M → 22.

Common environment: `QUELL_ROOT=~/quell-edge-llm-ids`. `QUELL_FEATSET=noid` selects the identifier-free Edge-IIoTset
feature set (45 features; 11 identifier fields removed, `quell_common.NOID_DROP`, whose `tcp.stream` entry is not a column of the dataset); empty = all features.

## 1. Detection quality (Table III, §IV-A, §III-I)

| result file | script · settings | log | manuscript |
|---|---|---|---|
| `llm_report_fullctx_Qwen2.5-1.5B_s42_noid.json`, `_s1_noid`, `_s2_noid` | `12_detection.py` · `QUELL_MODEL=Qwen/Qwen2.5-1.5B QUELL_DATASETS=edge_iiotset QUELL_SEED={42,1,2} QUELL_FEATSET=noid` | `12_detection_q15_noid`, `_q15_noid_s1`, `_q15_noid_s2` | Table III (Edge-IIoTset 0.892 ± 0.016) |
| `llm_report_fullctx_Qwen2.5-1.5B_s1.json`, `_s2.json` | `12_detection.py` · `QUELL_MODEL=Qwen/Qwen2.5-1.5B QUELL_DATASETS=ciciot2023,nbaiot QUELL_SEED={1,2}` | `12_detection_q15_s1`, `_q15_s2` | Table III (CICIoT2023, N-BaIoT; seed 42 = `results/llm_report_fullctx.json`) |
| `llm_report_fullctx_Qwen2.5-0.5B_s42.json` | `12_detection.py` · `QUELL_MODEL=Qwen/Qwen2.5-0.5B QUELL_DATASETS=edge_iiotset,ciciot2023,nbaiot QUELL_SEED=42` | `12_detection_q05` | §IV-A (CICIoT2023 0.834, N-BaIoT 0.803) |
| `llm_report_fullctx_Qwen2.5-0.5B_s42_noid.json` | as above with `QUELL_DATASETS=edge_iiotset QUELL_FEATSET=noid` | `12_detection_q05_noid` | §IV-A (Edge-IIoTset 0.901) |
| `llm_report_fullctx_Llama-3.2-1B_s42_noid.json`, `_s1_noid`, `_s2_noid` | `12_detection.py` · `QUELL_MODEL=unsloth/Llama-3.2-1B QUELL_DATASETS=edge_iiotset QUELL_SEED={42,1,2} QUELL_FEATSET=noid` | `12_detection_llama_noid_s42`, `_s1`, `_s2` | Table III (0.901 ± 0.006) |
| `llm_report_fullctx_Llama-3.2-1B_s42.json` | `12_detection.py` · `QUELL_MODEL=unsloth/Llama-3.2-1B QUELL_DATASETS=ciciot2023 QUELL_SEED=42` | `12_detection_llama_ciciot_s42` | Table III (CICIoT2023 0.828) |
| `leakage_check_edge_iiotset.json` | `10_leakage_check.py` (feature sets base / ferrag / noid / strict, seeds 42, 1, 2, 3, 4) | `10_leakage_check`, `10_leakage_check_noid` | Table III (XGBoost 0.895, RF 0.884; control row 0.987 / 0.971), §III-C |
| `token_stats.json` | `11_token_stats.py` | — | §III-D (median, 99th percentile, maximum length) |
| `derived_stats.json` | `21_derived_stats.py > results/v4/derived_stats.json` (reads only `results/`) | — | seed means and s.d., paired 95% CIs, Glass's Δ, N-BaIoT present-class macro-F1, Table VIII summaries |

`unsloth/Llama-3.2-1B` is an ungated, byte-identical mirror of `meta-llama/Llama-3.2-1B`.

## 2. Quantization and deployed export (Table IV, §III-G, §IV-B)

| result file | script · settings | log | manuscript |
|---|---|---|---|
| `quant_retention_edge_iiotset_Qwen2.5-1.5B_s42_noid.json` | `13_quantize_export.py --dataset edge_iiotset --seed 42 --adapter <exp>/adapter` · `QUELL_FEATSET=noid QUELL_MERGE=dequant` (`qlora_merge.py`) | `13_quantize_export-2_noid` | Table IV (FP16 0.8916 / INT8 0.8901 / NF4 0.8914) |
| `quant_retention_edge_iiotset_Qwen2.5-1.5B_s42_noid_plainmerge.json` | same, merge into the full-precision base (`QUELL_MERGE=plain`) | `13_quantize_export_noid` | superseded; §V-C pitfall (iv) (FP16 0.821) |
| `quant_retention_edge_iiotset_Qwen2.5-1.5B_s42.json`, `export_meta_edge_iiotset_Qwen2.5-1.5B_s42.json` | `13_quantize_export.py --dataset edge_iiotset --seed 42` (all features, `QUELL_MERGE=plain`) | `13_quantize_export` | all-feature control |
| `gguf_noid/gguf_eval_f16_ypb.json`, `gguf_noid/meta.json`, `gguf_noid/llama_cpp_commit.txt` | `14_gguf_make.sh <exp> ~/llama.cpp`, then `14_gguf_run.sh <exp> 100` (`14_gguf_eval.py`, last-token pooling + trained head) | `14_gguf-2_noid`, `14_gguf-3_noid_f16` | §III-G (GGUF F16 agrees with HF-FP16 on 99/100 flows; llama.cpp 4413a19) |
| `gguf_noid/gguf_eval_q8_0_ypb.json`, `gguf_noid/gguf_eval_q4_k_m_ypb.json` | placeholders; Q8_0 and Q4_K_M are measured on the Jetson (section 5) | — | — |
| `gguf/*` | `14_gguf_make.sh` + `14_gguf_run.sh <exp> 300` (all features) | `14_gguf`, `14_gguf_noid` | all-feature control; `meta_noid_plainmerge.json` = superseded plain-merge export |
| `jetson_bundle_summary.json` | `18_jetson_bundle.py` | `18_jetson_bundle` | contents of the bundle copied to the Jetson |

## 3. Unknown-attack generalization (Table VIII, Fig. 5)

| result file | script · settings | log | manuscript |
|---|---|---|---|
| `generalization_edge_iiotset_Qwen2.5-1.5B_s42_noid.json` | `15_generalization.py` · `QUELL_DATASET=edge_iiotset QUELL_FEATSET=noid` (leave-one-attack-out) | `15_generalization_edge_noid` | Table VIII (Edge-IIoTset) |
| `generalization_edge_iiotset_classic_s42_noid.json` | same, `QUELL_CLASSIC_ONLY=1` | `15_generalization_edge_noid_classic` | cross-check of the RF / XGBoost rows |
| `generalization_ciciot2023_Qwen2.5-1.5B_s42_family.json` | `15_generalization.py` · `QUELL_DATASET=ciciot2023` (7 families held out) | `15_generalization_ciciot` | Table VIII, Fig. 5 |
| `generalization_nbaiot_Qwen2.5-1.5B_s42_family.json` | `15_generalization.py` · `QUELL_DATASET=nbaiot QUELL_FEATSET=` (gafgyt / mirai held out) | `15_generalization_nbaiot` | Table VIII, Fig. 5 |

## 4. Robustness (Tables IX and X, Fig. 6)

| result file | script · settings | log | manuscript |
|---|---|---|---|
| `adversarial_edge_iiotset_Qwen2.5-1.5B_s42_noid.json` | `16_adversarial.py` · `QUELL_DATASET=edge_iiotset QUELL_FEATSET=noid` | `16_adversarial_edge_noid` | Table IX, Fig. 6 |
| `adversarial_ciciot2023_Qwen2.5-1.5B_s42.json` | `16_adversarial.py` · `QUELL_DATASET=ciciot2023` | `16_adversarial_ciciot` | Table IX, Fig. 6 |
| `adversarial_nbaiot_classic_s42.json` | `16_adversarial.py` · `QUELL_DATASET=nbaiot QUELL_CLASSIC_ONLY=1` | `16_adversarial_nbaiot_classic` | Table IX, Fig. 6 (LLM row from `results/adversarial_report_nbaiot_fullctx.json`, identical protocol) |
| `adversarial_edge_iiotset_classic_s42.json`, `adversarial_edge_iiotset_classic_s42_noid.json` | `16_adversarial.py` · `QUELL_DATASET=edge_iiotset QUELL_CLASSIC_ONLY=1` (± `QUELL_FEATSET=noid`) | `16_adversarial_edge_classic`, `16_adversarial_edge_noid_classic` | cross-check (LLM fields empty by design) |
| `constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid.json` | `17_constrained_attack.py` · `QUELL_DATASET=edge_iiotset QUELL_FEATSET=noid QUELL_PRECISIONS=fp16,int8` (13 controllable fields, projection, HopSkipJump at 0.1 / 0.3 / 1.0σ); INT8 HopSkipJump with `QUELL_PRECISIONS=int8 QUELL_HSJ_TIMEOUT=240` | `17_constrained_attack-2_edge_noid`, `17_constrained_attack-3_int8` | Table X, Fig. 6 (format-controlled LLM curve) |
| `constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid_int8run1.json` | as above, first INT8 HopSkipJump run (its log was overwritten; all other entries identical to the main file) | — | Table X note (INT8 run-to-run variation, 13% vs. 17%) |
| `constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid_plainmerge.json` | as above on the plain-merge export | `17_constrained_attack_edge_noid` | superseded |

Robustness runs require `adversarial-robustness-toolbox==1.20.1`.

## 5. On-device measurements (Jetson Orin Nano Super) — `results/v4/jetson/`, `logs/v4/jetson/`

Code: `19_jetson_run.sh` (calls `14_gguf_eval.py` and `19_jetson_classic_cost.py`) on the bundle built by
`18_jetson_bundle.py`; llama.cpp built on the device at the export commit 4413a19. Conditions: MAXN_SUPER
(`nvpmodel -m 2`), `jetson_clocks`, desktop stopped (`systemctl isolate multi-user.target`). The N-BaIoT length set
was run separately because the board rebooted twice under that sustained load (single pass with ubatch 1728; chunked
prefill); its prefill times are parsed from `server_q8_0_nbaiot_chunked.log.gz` (179 flows before the reboot).
`tegra_watch.log.gz` = temperature and power every 5 s.

Summary: `python scripts/20_jetson_summary.py > results/v4/jetson/jetson_summary.json`.
Manuscript: Tables VI and VII, Fig. 4, §III-G, §IV-C, §V-D. Table V is the earlier FP16 measurement
(`results/jetson_fp16.json`).

## 6. Tables

`tables/table02_datasets.csv` … `tables/table11_verdict.csv` = Tables II–XI of the manuscript, written by
`23_make_tables.py` from the files above (every cell computed, formatted as printed).

## 7. Software versions

GPU host: Adversarial Robustness Toolbox 1.20.1, scikit-learn 1.9.0, XGBoost 3.3.0, PyTorch 2.11.0+cu128,
Transformers 5.14.1. Jetson: JetPack 6.2 (L4T R36.4.7), llama.cpp 4413a19 (CUDA, sm_87), scikit-learn 1.7.2,
XGBoost 3.2.0, ONNX Runtime 1.23.2.

## 8. Earlier results

`results/*.json` are the results of the earlier versions: the original runs and the full-context re-runs
(`*_fullctx.json`, `scripts/re_*.py`). They are kept unchanged; the manuscript uses them only where stated above.
