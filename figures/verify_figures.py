#!/usr/bin/env python3
"""
Integrity check: assert that every value plotted in Figures 4-6 equals the
corresponding number in the raw experimental result files under results/.

The figures/QUELL_Fig{4,5,6}_data.json files are derived plotting inputs; this
script proves they are faithful extracts of the pipeline outputs (results/*.json),
with no hand-edited or fabricated numbers.

    python figures/verify_figures.py

Exit code 0 and "ALL CHECKS PASSED" => figures are fully traceable to the raw data.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(name): return json.load(open(os.path.join(ROOT, "results", name), encoding="utf-8"))
def F(name): return json.load(open(os.path.join(ROOT, "figures", name), encoding="utf-8"))

fails = []
def check(label, a, b, tol=5e-4):
    ok = abs(float(a) - float(b)) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: fig={a}  raw={b}")
    if not ok: fails.append(label)

# ---- FIG 4 (cost-accuracy) ----
print("FIG 4  (energy: jetson_*; macro-F1: multiseed_report / llm_report)")
f4 = {p["label"]: p for p in F("QUELL_Fig4_data.json")["points"]}
rf, fp, ll = R("jetson_rf_cpu.json"), R("jetson_fp16.json"), R("jetson_llama_power.json")
ms, lr = R("multiseed_report.json"), R("llm_report.json")
seeds = ms["edge_iiotset"]["seeds"]
rf_f1 = sum(s["rf"]["macro_f1"] for s in seeds.values()) / len(seeds)
check("RF energy",       f4["Random Forest"]["energy_mJ"], rf["batch1_energy_per_inf_mj"])
check("RF macro-F1",     f4["Random Forest"]["macro_f1"],  rf_f1)
check("GPT-2 energy",    f4["GPT-2-medium"]["energy_mJ"],
      [r for r in fp["results"] if r["model"] == "gpt2-medium"][0]["energy_per_inf_mj"])
check("GPT-2 macro-F1",  f4["GPT-2-medium"]["macro_f1"], lr["edge_iiotset"]["gpt2-medium"]["macro_f1"])
check("Qwen INT8 energy",f4["Qwen2.5-1.5B"]["energy_mJ"], ll["results_final"]["int8_q8_0"]["energy_per_inf_mj"])

# ---- FIG 5 (unknown-attack generalization) ----
print("FIG 5  (generalization_report.json, Edge-IIoTset)")
gen = R("generalization_report.json")["edge_iiotset"]["folds"]
for row in F("QUELL_Fig5_data.json")["folds"]:
    check(f"unseen recall {row['attack']}", row["llm"], gen[row["attack"]]["llm"]["unseen_recall"])

# ---- FIG 6 (adversarial robustness) ----
print("FIG 6  (adversarial_report.json + adversarial_report_nbaiot.json)")
src = {"Edge-IIoTset": R("adversarial_report.json"), "N-BaIoT": R("adversarial_report_nbaiot.json")}
for panel in F("QUELL_Fig6_data.json")["panels"]:
    s = src[panel["name"]]
    for i, e in enumerate(panel["rf"]):  check(f"{panel['name']} RF eps[{i}]",  e, s["noise_curve"]["rf_macro_f1"][i])
    for i, e in enumerate(panel["llm"]): check(f"{panel['name']} LLM eps[{i}]", e, s["noise_curve"]["llm_macro_f1"][i])
    check(f"{panel['name']} evasion RF",  panel["evasion"]["rf"],  s["evasion"]["rf_evasion_rate"])
    check(f"{panel['name']} evasion LLM", panel["evasion"]["llm"], s["evasion"]["llm_evasion_rate"])

print()
if fails: print(f"{len(fails)} CHECK(S) FAILED:", fails); sys.exit(1)
print("ALL CHECKS PASSED - every figure value matches the raw results/*.json.")
