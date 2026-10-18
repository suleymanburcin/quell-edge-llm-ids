#!/usr/bin/env python3
"""QUELL Step 21 (any machine, seconds) — every derived number quoted in the revised manuscript that is not read
directly from a single result file: seed means/s.d., paired 95% CIs and Glass's Delta (Section III-I, IV-A),
N-BaIoT macro-F1 over the classes present in the test set (Table III note), generalization summaries
(Table VIII) and the quantization deltas (Section IV-B). Reads only results/ and prints a JSON report.
All s.d. are sample s.d. (ddof = 1); leakage_check stores population s.d. (kept as sd_5seeds_population_as_stored).
Usage:  python scripts/21_derived_stats.py  [> results/v4/derived_stats.json]
"""
import json, os, sys
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = lambda *p: json.load(open(os.path.join(R, *p)))
out = {}

# Edge-IIoTset, identifier-free: LLM seeds 42/1/2 vs RF/XGB seeds (leakage_check, matched protocol)
llm = {s: J("v4", f"llm_report_fullctx_Qwen2.5-1.5B_s{s}_noid.json")["edge_iiotset"]["Qwen/Qwen2.5-1.5B"]["macro_f1"] for s in (42, 1, 2)}
lk = J("v4", "leakage_check_edge_iiotset.json")
L = np.array([llm[s] for s in (42, 1, 2)])
e = dict(llm_seeds=llm, llm_mean=round(L.mean(), 4), llm_sd=round(L.std(ddof=1), 4))
for m in ("rf", "xgb"):
    sd = dict(zip(lk["seeds"], lk["results"]["noid"][m]["seeds"])); C = np.array([sd[s] for s in (42, 1, 2)]); d = L - C
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    e[m] = dict(mean_5seeds=lk["results"]["noid"][m]["mean"], sd_5seeds=round(float(np.std(lk["results"]["noid"][m]["seeds"], ddof=1)), 4),
                sd_5seeds_population_as_stored=lk["results"]["noid"][m]["std"],
                paired_diff_llm_minus=round(d.mean(), 4), ci95=[round(d.mean() - h, 3), round(d.mean() + h, 3)],
                glass_delta=round((np.mean(lk["results"]["noid"][m]["seeds"]) - L.mean()) / L.std(ddof=1), 2))
out["edge_noid_detection"] = e

# CICIoT2023 / N-BaIoT LLM seeds (seed 42 = results/llm_report_fullctx.json)
s42 = J("llm_report_fullctx.json")
for ds in ("ciciot2023", "nbaiot"):
    v = [s42[ds]["Qwen/Qwen2.5-1.5B"]] + [J("v4", f"llm_report_fullctx_Qwen2.5-1.5B_s{s}.json")[ds]["Qwen/Qwen2.5-1.5B"] for s in (1, 2)]
    f = np.array([x["macro_f1"] for x in v]); r = dict(seeds=f.tolist(), mean=round(f.mean(), 4), sd=round(f.std(ddof=1), 4))
    if ds == "nbaiot":   # mirai_ack / mirai_udp have no test flows -> macro-F1 over the 9 present classes
        pres = [np.mean([a for k, a in x["per_class_f1"].items() if k not in ("mirai_ack", "mirai_udp")]) for x in v]
        r.update(present_class=[round(p, 4) for p in pres], present_mean=round(np.mean(pres), 4), present_sd=round(np.std(pres, ddof=1), 4),
                 classical_present_from_all_class=dict(rf=round(0.817 * 11 / 9, 3), xgb=round(0.816 * 11 / 9, 3)))
    out[f"{ds}_llm_3seeds"] = r

# Generalization summaries (Table VIII)
g = {}
for ds, f in (("edge_iiotset", "generalization_edge_iiotset_Qwen2.5-1.5B_s42_noid.json"),
              ("ciciot2023", "generalization_ciciot2023_Qwen2.5-1.5B_s42_family.json"),
              ("nbaiot", "generalization_nbaiot_Qwen2.5-1.5B_s42_family.json")):
    folds = J("v4", f)["folds"]
    g[ds] = {m: dict(mean_recall=round(np.mean([v[m]["unseen_recall"] for v in folds.values()]), 4),
                     min_recall=round(min(v[m]["unseen_recall"] for v in folds.values()), 4),
                     mean_benign_fpr=round(np.mean([v[m]["benign_fpr"] for v in folds.values()]), 4),
                     per_fold={k: v[m]["unseen_recall"] for k, v in folds.items()}) for m in ("rf", "xgb", "llm")}
out["generalization"] = g

# Quantization (Table IV, faithful merge)
q = J("v4", "quant_retention_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")["precisions"]
out["quantization_noid"] = {p: q[p]["macro_f1_all"] for p in q} | {
    "delta_int8": round(q["int8"]["macro_f1_all"] - q["fp16"]["macro_f1_all"], 4),
    "delta_nf4": round(q["nf4"]["macro_f1_all"] - q["fp16"]["macro_f1_all"], 4)}
json.dump(out, sys.stdout, indent=2, default=float)
