#!/usr/bin/env python3
"""QUELL Step 20 (any machine) — one summary of the on-device (Jetson Orin Nano Super, MAXN_SUPER, clocks pinned,
desktop stopped) measurements of 2026-10-17, read from results/v4/jetson/ and logs/v4/jetson/.
* deployed LLM detector (GGUF Q8_0 / Q4_K_M + trained head): on-device macro-F1 and agreement with HF-FP16 (1,000
  stratified Edge-IIoTset flows), batch-1 latency, board power (VDD_IN), energy, RAM — at the Edge-IIoTset and
  CICIoT2023 prompt lengths (single pass, ubatch 512)
* N-BaIoT prompt length (~1,460 tokens): the board rebooted twice under this sustained load (single pass with ubatch
  1728, then chunked prefill); the server-side prompt-evaluation times of the flows processed before the second reboot
  (Q8_0, 512-token chunks) are read from the llama-server log. No power/energy for this set.
* trained RF / XGBoost on the CPU (scikit-learn, XGBoost native, ONNX Runtime)
Usage: python scripts/20_jetson_summary.py > results/v4/jetson/jetson_summary.json
"""
import gzip, json, os, re, statistics as st, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
J = lambda f: json.load(open(os.path.join(R, "results", "v4", "jetson", f)))
out = {"conditions": dict(device="Jetson Orin Nano Engineering Reference Developer Kit Super (8 GB)", power_mode="MAXN_SUPER (nvpmodel 2), jetson_clocks",
                          desktop="stopped (multi-user.target)", llama_cpp=open(os.path.join(R, "results/v4/jetson/llama_cpp_commit_jetson.txt")).read().strip(),
                          energy="board input power (VDD_IN, tegrastats 100 ms) x mean batch-1 latency; includes idle board power")}
llm = {}
for t in ("q8_0", "q4_k_m"):
    a = J(f"gguf_eval_{t}_jetson.json")
    llm[t] = dict(macro_f1=a["metrics"]["macro_f1_all"], hf_fp16_same_rows=a["hf_fp16_on_same_rows"]["macro_f1_all"],
                  agreement_with_hf_fp16=a["agreement_with_hf_fp16"], n=a["n"])
    for ds in ("edge_iiotset", "ciciot2023"):
        c = J(f"gguf_eval_{t}_jetson_len-{ds}_ids.json")
        llm[t][ds] = dict(latency_ms=c["latency_ms"], tokens_mean=c["prompt_tokens"]["mean"], power_w=c["power"]["mean_w"],
                          energy_mj=c["energy_mj_per_inference"], ram_peak_mb=c["power"]["ram_peak_mb"], n=c["n"])
txt = gzip.open(os.path.join(R, "logs/v4/jetson/server_q8_0_nbaiot_chunked.log.gz"), "rt").read()
P = [(float(a), int(b)) for a, b in re.findall(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", txt)][5:]  # drop 5 warm-up requests
llm["q8_0"]["nbaiot_chunked_server_side"] = dict(n=len(P), prefill_ms_mean=round(st.mean(p[0] for p in P), 1),
    prefill_ms_median=round(st.median(p[0] for p in P), 1), tokens_mean=round(st.mean(p[1] for p in P), 1),
    note="server-side prompt evaluation (no HTTP overhead), 512-token chunks; run interrupted by a board reboot")
out["llm"] = llm
c = J("classic_cost_jetson.json"); out["classic_versions"] = {k: c[k] for k in ("sklearn", "xgboost", "onnxruntime", "threads")}
out["classic"] = {ds: {m: dict(macro_f1=v["macro_f1"], batch1_ms=v["batch1_latency_ms"]["mean"], batched_ms_per_sample=v["batched_ms_per_sample"],
                               throughput_per_s=v["batched_throughput_per_s"], energy_mj_batch1=v["energy_mj_batch1"],
                               energy_mj_batched=v["energy_mj_batched_per_sample"]) for m, v in r.items() if isinstance(v, dict) and "macro_f1" in v}
                  for ds, r in c["datasets"].items()}
w = gzip.open(os.path.join(R, "logs/v4/jetson/tegra_watch.log.gz"), "rt").read()
out["thermal_watch_14_40_to_15_00"] = dict(gpu_c_max=max(float(x) for x in re.findall(r"gpu@([\d.]+)C", w)),
                                           vdd_in_mw_max=max(int(x) for x in re.findall(r"VDD_IN (\d+)mW", w)))
json.dump(out, sys.stdout, indent=2)
