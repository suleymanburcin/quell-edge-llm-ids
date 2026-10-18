#!/usr/bin/env python3
"""QUELL Step 23 (any machine, seconds) — regenerate Tables II–XI of the manuscript from the released result files.

Every cell is computed from results/ (no value is typed in) and formatted as printed in the manuscript; the tables are
written to results/v4/tables/ as CSV and printed to stdout. Table I compares published studies and is not derived from
results. Usage:  python scripts/23_make_tables.py
"""
import csv, json, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "v4", "tables")
J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))


def f(x, nd=3):
    """round half up (as printed in the manuscript)"""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pm(m, s, nd=3):
    return f"{f(m, nd)} ± {f(s, nd)}"


def thousands(x, nd=0):
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{v:,.{nd}f}"


def write(name, title, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    print(f"\n{title}  ->  results/v4/tables/{name}")
    for r in rows:
        print("  " + " | ".join(r))


sp = J("split_report.json"); tok = J("v4", "token_stats.json")
lk = J("v4", "leakage_check_edge_iiotset.json")["results"]; ds = J("v4", "derived_stats.json")
bm = J("baseline_matched_report.json"); ms = J("multiseed_report_fullctx.json")["edge_iiotset"]["seeds"]
g2 = J("llm_report.json"); js = J("v4", "jetson", "jetson_summary.json")
l8, l4 = js["llm"]["q8_0"], js["llm"]["q4_k_m"]; ce = js["classic"]["edge_iiotset"]

# ---------------------------------------------------------------- Table II
names = {"edge_iiotset": "Edge-IIoTset", "ciciot2023": "CICIoT2023", "nbaiot": "N-BaIoT"}
strategy = {"edge_iiotset": "per-class temporal", "ciciot2023": "stratified + exact dedup",
            "nbaiot": "session-grouped (source file)"}
rows = [["Dataset", "Records / features / classes", "Benign share", "Split strategy"]]
for k in ("edge_iiotset", "ciciot2023", "nbaiot"):
    d = sp[k]["class_dist_overall"]; b = next(c for c in d if c.lower() in ("normal", "benigntraffic", "benign"))
    rows.append([names[k], f"{thousands(sp[k]['n_rows'])} / {tok[k]['n_feat']} / {sp[k]['n_classes']}",
                 f"{b} {f(100 * d[b] / sum(d.values()), 1)}%", strategy[k]])
write("table02_datasets.csv", "TABLE II  Datasets after preprocessing", rows)

# ---------------------------------------------------------------- Table III
LL = [J("v4", f"llm_report_fullctx_Llama-3.2-1B_s{s}_noid.json")["edge_iiotset"]["unsloth/Llama-3.2-1B"]["macro_f1"]
      for s in (42, 1, 2)]
ctrl = [s["llm"]["macro_f1"] for s in ms.values()]
e = ds["edge_noid_detection"]
sd = lambda r: st.stdev(r["seeds"])      # sample s.d. over the stored per-seed values
rows = [["Dataset", "XGBoost", "Random Forest", "GPT-2-medium", "Qwen2.5-1.5B", "Llama-3.2-1B"],
        ["Edge-IIoTset", pm(lk["noid"]["xgb"]["mean"], sd(lk["noid"]["xgb"])), pm(lk["noid"]["rf"]["mean"], sd(lk["noid"]["rf"])),
         "—", pm(e["llm_mean"], e["llm_sd"]), pm(st.mean(LL), st.stdev(LL))],
        ["Edge-IIoTset, with identifier fields", pm(lk["base"]["xgb"]["mean"], sd(lk["base"]["xgb"])),
         pm(lk["base"]["rf"]["mean"], sd(lk["base"]["rf"])), f(g2["edge_iiotset"]["gpt2-medium"]["macro_f1"]),
         pm(st.mean(ctrl), st.stdev(ctrl)), "—"],
        ["CICIoT2023", pm(bm["ciciot2023"]["xgb"]["mean"], sd(bm["ciciot2023"]["xgb"])),
         pm(bm["ciciot2023"]["rf"]["mean"], sd(bm["ciciot2023"]["rf"])), f(g2["ciciot2023"]["gpt2-medium"]["macro_f1"]),
         pm(ds["ciciot2023_llm_3seeds"]["mean"], ds["ciciot2023_llm_3seeds"]["sd"]),
         f(J("v4", "llm_report_fullctx_Llama-3.2-1B_s42.json")["ciciot2023"]["unsloth/Llama-3.2-1B"]["macro_f1"])],
        ["N-BaIoT", pm(bm["nbaiot"]["xgb"]["mean"], sd(bm["nbaiot"]["xgb"])), pm(bm["nbaiot"]["rf"]["mean"], sd(bm["nbaiot"]["rf"])),
         f(g2["nbaiot"]["gpt2-medium"]["macro_f1"]), pm(ds["nbaiot_llm_3seeds"]["mean"], ds["nbaiot_llm_3seeds"]["sd"]), "—"]]
write("table03_detection_macro_f1.csv", "TABLE III  Detection quality (macro-F1)", rows)

# ---------------------------------------------------------------- Table IV
q = J("v4", "quant_retention_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")["precisions"]
rows = [["Precision", "macro-F1", "Accuracy"]] + [[lab, f(q[p]["macro_f1_all"], 4), f(q[p]["acc"], 4)]
        for lab, p in (("FP16 (merged)", "fp16"), ("INT8", "int8"), ("4-bit (NF4)", "nf4"))]
write("table04_quant_retention.csv", "TABLE IV  Accuracy retention under quantization", rows)

# ---------------------------------------------------------------- Table V (earlier FP16 measurement)
fp = {r["model"]: r for r in J("jetson_fp16.json")["results"]}
rows = [["Model", "Size", "Lat. (ms)", "Peak mem (MB)", "Power (W)", "Energy (mJ)"]]
for lab, k in (("GPT-2-medium", "gpt2-medium"), ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B")):
    r = fp[k]
    rows.append([lab, f"{r['params_m']}M", f(r["latency_ms_avg"], 1), f(r["peak_gpu_mem_mb"], 0), f(r["avg_board_power_w"], 1),
                 f(r["energy_per_inf_mj"], 1)])
r = fp["Qwen/Qwen2.5-1.5B"]
rows.append(["Qwen2.5-1.5B", f"{f(r['params_m'] / 1000, 1)}B", "—", "did not fit" if r["status"] == "load_failed_oom" else "?", "—", "—"])
write("table05_jetson_fp16_cost.csv", "TABLE V  FP16 on-device cost (earlier measurement)", rows)

# ---------------------------------------------------------------- Table VI
size = {}
for line in open(os.path.join(R, "v4", "jetson", "gguf_sizes.txt"), encoding="utf-8"):
    p = line.split()
    if len(p) > 4 and p[-1].endswith(".gguf"):
        size[p[-1].rsplit("model-", 1)[-1][:-5]] = int(p[4]) / 2 ** 30
le, lc, nb = l8["edge_iiotset"], l8["ciciot2023"], l8["nbaiot_chunked_server_side"]
rows = [["", "INT8 (Q8_0)", "4-bit (Q4_K_M)"],
        ["Model size", f"{f(size['q8_0'], 2)} GiB", f"{f(size['q4_k_m'], 2)} GiB"],
        ["Macro-F1 (agreement with HF-FP16)", f"{f(l8['macro_f1'])} ({f(100 * l8['agreement_with_hf_fp16'], 1)}%)",
         f"{f(l4['macro_f1'])} ({f(100 * l4['agreement_with_hf_fp16'], 1)}%)"]]
for lab, k in (("Edge-IIoTset", "edge_iiotset"), ("CICIoT2023", "ciciot2023")):
    rows.append([f"{lab}, {f(l8[k]['tokens_mean'], 0)} tokens",
                 f"{f(l8[k]['latency_ms']['mean'], 0)} ms / {f(l8[k]['energy_mj'] / 1000, 2)} J",
                 f"{f(l4[k]['latency_ms']['mean'], 0)} ms / {f(l4[k]['energy_mj'] / 1000, 2)} J"])
rows.append([f"N-BaIoT, {thousands(nb['tokens_mean'])} tokens", f"{f(nb['prefill_ms_mean'], 0)} ms*", "—"])
write("table06_deployed_detector.csv", "TABLE VI  On-device accuracy and cost of the deployed detector", rows)

# ---------------------------------------------------------------- Table VII
ms_llm = le["latency_ms"]["mean"]; mj_llm = le["energy_mj"]; fps_llm = 1000 / ms_llm
def mj(x): return f(x, 0) if x >= 100 else (f(x, 1) if x >= 10 else f(x, 2))
def ms_(x): return f(x, 1) if x >= 10 else (f(x, 2) if x >= 1 else f(x, 2))
rf_, xg = f(lk["noid"]["rf"]["mean"]), f(lk["noid"]["xgb"]["mean"])
rnd = lambda x, step: thousands(round(x / step) * step)
rows = [["Detector (runtime)", "Macro-F1", "Latency", "Energy", "Flows/s"],
        ["Random forest (scikit-learn)", rf_, f"{ms_(ce['rf_sklearn']['batch1_ms'])} ms", f"{mj(ce['rf_sklearn']['energy_mj_batch1'])} mJ", thousands(ce["rf_sklearn"]["throughput_per_s"])],
        ["Random forest (ONNX Runtime)", rf_, f"{ms_(ce['rf_onnxruntime']['batch1_ms'])} ms", f"{mj(ce['rf_onnxruntime']['energy_mj_batch1'])} mJ", thousands(ce["rf_onnxruntime"]["throughput_per_s"])],
        ["XGBoost (native)", xg, f"{ms_(ce['xgb_native']['batch1_ms'])} ms", f"{mj(ce['xgb_native']['energy_mj_batch1'])} mJ", thousands(ce["xgb_native"]["throughput_per_s"])],
        ["LLM INT8 (llama.cpp, GPU)", f(e["llm_mean"]), f"{f(ms_llm, 0)} ms", f"{thousands(mj_llm)} mJ", f(fps_llm, 1)],
        ["LLM vs. best classical", "—", f"≈{rnd(ms_llm / ce['rf_onnxruntime']['batch1_ms'], 1000)}×",
         f"≈{rnd(mj_llm / ce['rf_onnxruntime']['energy_mj_batch1'], 1000)}×",
         "—"]]
write("table07_cost_same_jetson.csv", "TABLE VII  Cost of the deployed detectors on the same Jetson", rows)

# ---------------------------------------------------------------- Table VIII
g = ds["generalization"]; M = ("rf", "xgb", "llm")
rows = [["Held-out unit / metric", "RF", "XGBoost", "LLM"],
        ["Edge-IIoTset, 14 classes: mean recall"] + [f(g["edge_iiotset"][m]["mean_recall"]) for m in M],
        ["lowest fold"] + [f(g["edge_iiotset"][m]["min_recall"]) for m in M],
        ["benign FPR (mean)"] + [f(g["edge_iiotset"][m]["mean_benign_fpr"]) for m in M],
        ["CICIoT2023, 7 families: mean recall"] + [f(g["ciciot2023"][m]["mean_recall"]) for m in M],
        ["lowest family"] + [f(g["ciciot2023"][m]["min_recall"]) for m in M],
        ["benign FPR (mean)"] + [f(g["ciciot2023"][m]["mean_benign_fpr"]) for m in M],
        ["N-BaIoT: gafgyt held out"] + [f(g["nbaiot"][m]["per_fold"]["gafgyt"]) for m in M],
        ["mirai held out"] + [f(g["nbaiot"][m]["per_fold"]["mirai"]) for m in M],
        ["benign FPR (mean)"] + [f(g["nbaiot"][m]["mean_benign_fpr"]) for m in M]]
write("table08_unknown_attack.csv", "TABLE VIII  Unknown-attack detection", rows)

# ---------------------------------------------------------------- Table IX
ae = J("v4", "adversarial_edge_iiotset_Qwen2.5-1.5B_s42_noid.json"); ac = J("v4", "adversarial_ciciot2023_Qwen2.5-1.5B_s42.json")
an = J("v4", "adversarial_nbaiot_classic_s42.json"); al = J("adversarial_report_nbaiot_fullctx.json")
rows = [["Detector", "ε = 0", "0.1", "0.25", "0.5", "1.0", "Evasion"]]
for lab, src in (("Edge-IIoTset", ae), ("CICIoT2023", ac), ("N-BaIoT", None)):
    for m, name in (("rf", "RF"), ("xgb", "XGBoost"), ("llm", "LLM")):
        s = src if src is not None else (al if m == "llm" else an)
        rows.append([f"{name} ({lab})"] + [f(v) for v in s["noise_curve"][f"{m}_macro_f1"]] + [f(s["evasion"][f"{m}_evasion_rate"])])
write("table09_random_noise.csv", "TABLE IX  Random-noise sensitivity and random-search evasion", rows)

# ---------------------------------------------------------------- Table X
cx = J("v4", "constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")
rows = [["Detector", "Clean", "All num. ε=0.1", "Ctrl. ε=0.1", "HSJ ≤1σ", "HSJ any"]]
for k, lab in (("rf", "RF"), ("xgb", "XGBoost"), ("llm_fp16", "LLM FP16"), ("llm_int8", "LLM INT8"), ("llm_nf4", "LLM NF4")):
    a, c, h = cx["noise"][k]["all_numeric_valid"]["macro_f1"], cx["noise"][k]["controllable_valid"]["macro_f1"], cx["hsj"][k]
    rows.append([lab, f(a[0]), f(a[1]), f(c[1]), f(h["success_at_budget"]["1.0"], 2), f(h["success_unbounded"], 2)])
write("table10_constrained.csv", "TABLE X  Attacker-constrained, format-controlled evaluation", rows)

# ---------------------------------------------------------------- Table XI
n2 = lambda x: f(x, 2)
gen = lambda k: "/".join(n2(g[k][m]["mean_recall"]) for m in M)
evs = [cx["hsj"][k]["success_at_budget"]["1.0"] for k in ("rf", "xgb", "llm_fp16", "llm_int8", "llm_nf4")]
q8drop = abs(l8["macro_f1"] - l8["hf_fp16_same_rows"]); q4drop = l4["macro_f1"] - l4["hf_fp16_same_rows"]
qws = max(abs(q[p]["macro_f1_all"] - q["fp16"]["macro_f1_all"]) for p in ("int8", "nf4"))
rows = [["Axis", "Better detector", "Evidence"],
        ["Clean accuracy (Edge-IIoTset)", "None",
         f"RF {n2(lk['noid']['rf']['mean'])} / XGB {n2(lk['noid']['xgb']['mean'])} vs LLM {n2(e['llm_mean'])} ± {n2(e['llm_sd'])} "
         f"(Llama-3.2-1B {n2(st.mean(LL))} ± {n2(st.stdev(LL))}); 95% CIs include 0"
         if e["rf"]["ci95"][0] < 0 < e["rf"]["ci95"][1] and e["xgb"]["ci95"][0] < 0 < e["xgb"]["ci95"][1] else "CHECK"],
        ["Clean accuracy (CICIoT, N-BaIoT)", "Mixed",
         f"LLM {n2(ds['ciciot2023_llm_3seeds']['mean'])} vs {n2(bm['ciciot2023']['xgb']['mean'])}/{n2(bm['ciciot2023']['rf']['mean'])} (CICIoT); "
         f"classical {n2(bm['nbaiot']['rf']['mean'])} vs {n2(ds['nbaiot_llm_3seeds']['mean'])} (N-BaIoT)"],
        ["Seed stability (Edge-IIoTset)", "Classical", f"s.d. {f(e['rf']['sd_5seeds'])}/{f(e['xgb']['sd_5seeds'])} vs {f(e['llm_sd'])}"],
        ["Unknown-attack generalization", "None consistent",
         f"RF/XGB/LLM: Edge {gen('edge_iiotset')}; CICIoT {gen('ciciot2023')}; N-BaIoT {gen('nbaiot')}"],
        ["Random-noise sensitivity", "Classical",
         f"ε = 0.1 on Edge: {n2(ae['noise_curve']['rf_macro_f1'][1])}/{n2(ae['noise_curve']['xgb_macro_f1'][1])} vs "
         f"{n2(ae['noise_curve']['llm_macro_f1'][1])} ({n2(cx['noise']['llm_fp16']['all_numeric_valid']['macro_f1'][1])} format-controlled); "
         "LLM lowest on all datasets"],
        ["Constrained evasion (HSJ, ≤1σ)", "None", f"{f(100 * max(evs), 0)}% for all detectors"],
        ["Quantization", "—",
         f"within {math.ceil(max(qws, q8drop) * 1000) / 1000} macro-F1 of FP16, also on the device (INT8); post-training 4-bit "
         f"−{n2(-q4drop)} on the device; robustness unchanged"],
        ["On-device cost (latency/energy)", "Classical",
         f"≈{rnd(ms_llm / ce['xgb_native']['batch1_ms'], 10)}× (XGBoost) to ≈{rnd(ms_llm / ce['rf_onnxruntime']['batch1_ms'], 1000)}× (RF, ONNX) latency; "
         f"≈{rnd(mj_llm / ce['xgb_native']['energy_mj_batch1'], 5)}–{rnd(mj_llm / ce['rf_onnxruntime']['energy_mj_batch1'], 1000)}× energy"]]
write("table11_verdict.csv", "TABLE XI  Axis-by-axis verdict", rows)
