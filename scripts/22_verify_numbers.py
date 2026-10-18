#!/usr/bin/env python3
"""QUELL Step 22 (any machine, seconds) — checks that the numbers printed in the revised manuscript's tables equal
the released result files. Every check names the table/section, the value as printed and the file it comes from.
Usage:  python scripts/22_verify_numbers.py        (exit code 0 and 'ALL n CHECKS PASSED' on success)
"""
import json, os, statistics as st, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))
fails, n = [], 0
def chk(where, printed, raw, tol=None):
    global n; n += 1
    tol = tol if tol is not None else 0.5 * 10 ** -(len(str(printed).split(".")[1]) if "." in str(printed) else 0) + 1e-9
    ok = abs(float(printed) - float(raw)) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {where}: printed {printed}  file {raw}")
    if not ok: fails.append(where)

lk = J("v4", "leakage_check_edge_iiotset.json")["results"]; ds = J("v4", "derived_stats.json")
bm = J("baseline_matched_report.json"); ms = J("multiseed_report_fullctx.json")["edge_iiotset"]["seeds"]
print("Table III — detection quality")
chk("III Edge noid XGBoost", 0.895, lk["noid"]["xgb"]["mean"]); chk("III Edge noid RF", 0.884, lk["noid"]["rf"]["mean"])
chk("III Edge noid LLM mean", 0.892, ds["edge_noid_detection"]["llm_mean"]); chk("III Edge noid LLM s.d.", 0.016, ds["edge_noid_detection"]["llm_sd"])
chk("III control XGBoost", 0.987, lk["base"]["xgb"]["mean"]); chk("III control RF", 0.971, lk["base"]["rf"]["mean"])
L = [s["llm"]["macro_f1"] for s in ms.values()]; chk("III control LLM mean (5 seeds)", 0.933, st.mean(L)); chk("III control LLM s.d.", 0.046, st.stdev(L))
chk("III CICIoT XGBoost", 0.762, bm["ciciot2023"]["xgb"]["mean"]); chk("III CICIoT RF", 0.748, bm["ciciot2023"]["rf"]["mean"])
chk("III CICIoT LLM mean", 0.829, ds["ciciot2023_llm_3seeds"]["mean"]); chk("III CICIoT LLM s.d.", 0.005, ds["ciciot2023_llm_3seeds"]["sd"])
chk("III N-BaIoT XGBoost", 0.816, bm["nbaiot"]["xgb"]["mean"]); chk("III N-BaIoT RF", 0.817, bm["nbaiot"]["rf"]["mean"])
chk("III N-BaIoT LLM mean", 0.794, ds["nbaiot_llm_3seeds"]["mean"]); chk("III N-BaIoT LLM s.d.", 0.026, ds["nbaiot_llm_3seeds"]["sd"])
chk("III note N-BaIoT present-class LLM", 0.970, ds["nbaiot_llm_3seeds"]["present_mean"]); chk("III note present-class s.d.", 0.032, ds["nbaiot_llm_3seeds"]["present_sd"])
sd = lambda r: st.stdev(r["seeds"])   # sample s.d. over the stored per-seed values
for where, v, r in (("III Edge noid XGBoost s.d.", "0.002", lk["noid"]["xgb"]), ("III Edge noid RF s.d.", "0.001", lk["noid"]["rf"]),
                    ("III control XGBoost s.d.", "0.002", lk["base"]["xgb"]), ("III control RF s.d.", "0.000", lk["base"]["rf"]),
                    ("III CICIoT XGBoost s.d.", "0.005", bm["ciciot2023"]["xgb"]), ("III CICIoT RF s.d.", "0.005", bm["ciciot2023"]["rf"]),
                    ("III N-BaIoT XGBoost s.d.", "0.000", bm["nbaiot"]["xgb"]), ("III N-BaIoT RF s.d.", "0.000", bm["nbaiot"]["rf"])):
    chk(where, v, sd(r))
g = J("llm_report.json")
chk("III GPT-2 control", 0.822, g["edge_iiotset"]["gpt2-medium"]["macro_f1"]); chk("III GPT-2 CICIoT", 0.475, g["ciciot2023"]["gpt2-medium"]["macro_f1"]); chk("III GPT-2 N-BaIoT", 0.618, g["nbaiot"]["gpt2-medium"]["macro_f1"])
q05 = J("v4", "llm_report_fullctx_Qwen2.5-0.5B_s42.json"); q05n = J("v4", "llm_report_fullctx_Qwen2.5-0.5B_s42_noid.json")
chk("IV-A Qwen-0.5B Edge noid", 0.901, q05n["edge_iiotset"]["Qwen/Qwen2.5-0.5B"]["macro_f1"])
chk("IV-A Qwen-0.5B CICIoT", 0.834, q05["ciciot2023"]["Qwen/Qwen2.5-0.5B"]["macro_f1"]); chk("IV-A Qwen-0.5B N-BaIoT", 0.803, q05["nbaiot"]["Qwen/Qwen2.5-0.5B"]["macro_f1"])
LL = [J("v4", f"llm_report_fullctx_Llama-3.2-1B_s{s}_noid.json")["edge_iiotset"]["unsloth/Llama-3.2-1B"]["macro_f1"] for s in (42, 1, 2)]
chk("III Llama-3.2-1B Edge noid mean (3 seeds)", 0.901, st.mean(LL)); chk("III Llama-3.2-1B Edge noid s.d.", 0.006, st.stdev(LL))
chk("III Llama-3.2-1B CICIoT", 0.828, J("v4", "llm_report_fullctx_Llama-3.2-1B_s42.json")["ciciot2023"]["unsloth/Llama-3.2-1B"]["macro_f1"])
for m, ci in (("rf", (-0.030, 0.046)), ("xgb", (-0.043, 0.036))):
    chk(f"IV-A paired CI LLM-{m} low", ci[0], ds["edge_noid_detection"][m]["ci95"][0]); chk(f"IV-A paired CI LLM-{m} high", ci[1], ds["edge_noid_detection"][m]["ci95"][1])
print("Table IV — quantization")
q = J("v4", "quant_retention_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")["precisions"]
for p, v in (("fp16", 0.8916), ("int8", 0.8901), ("nf4", 0.8914)): chk(f"IV {p} macro-F1", v, q[p]["macro_f1_all"])
print("Tables VI–VII, Fig. 4 — on device")
js = J("v4", "jetson", "jetson_summary.json"); l8, l4 = js["llm"]["q8_0"], js["llm"]["q4_k_m"]; ce = js["classic"]["edge_iiotset"]
chk("VI Q8_0 macro-F1", 0.907, l8["macro_f1"]); chk("VI Q8_0 agreement %", 99.1, 100 * l8["agreement_with_hf_fp16"]); chk("VI HF-FP16 same rows", 0.905, l8["hf_fp16_same_rows"])
chk("VI Q4_K_M macro-F1", 0.867, l4["macro_f1"]); chk("VI Q4_K_M agreement %", 94.2, 100 * l4["agreement_with_hf_fp16"])
chk("VI Q8_0 Edge ms", 278, l8["edge_iiotset"]["latency_ms"]["mean"]); chk("VI Q8_0 Edge J", 4.70, l8["edge_iiotset"]["energy_mj"] / 1000)
chk("VI Q8_0 CICIoT ms", 248, l8["ciciot2023"]["latency_ms"]["mean"]); chk("VI Q8_0 CICIoT J", 4.20, l8["ciciot2023"]["energy_mj"] / 1000)
chk("VI Q4_K_M Edge ms", 315, l4["edge_iiotset"]["latency_ms"]["mean"]); chk("VI Q4_K_M Edge J", 5.07, l4["edge_iiotset"]["energy_mj"] / 1000)
chk("VI Q4_K_M CICIoT ms", 279, l4["ciciot2023"]["latency_ms"]["mean"]); chk("VI Q4_K_M CICIoT J", 4.47, l4["ciciot2023"]["energy_mj"] / 1000)
chk("VI Q8_0 N-BaIoT prefill ms", 784, l8["nbaiot_chunked_server_side"]["prefill_ms_mean"])
chk("VII RF sklearn ms", 54.2, ce["rf_sklearn"]["batch1_ms"]); chk("VII RF sklearn mJ", 394, ce["rf_sklearn"]["energy_mj_batch1"])
chk("VII RF ONNX ms", 0.07, ce["rf_onnxruntime"]["batch1_ms"]); chk("VII RF ONNX mJ", 0.52, ce["rf_onnxruntime"]["energy_mj_batch1"])
chk("VII XGB ms", 2.53, ce["xgb_native"]["batch1_ms"]); chk("VII XGB mJ", 20.0, ce["xgb_native"]["energy_mj_batch1"])
chk("VII RF sklearn flows/s", 30854, ce["rf_sklearn"]["throughput_per_s"]); chk("VII RF ONNX flows/s", 89872, ce["rf_onnxruntime"]["throughput_per_s"])
chk("VII XGB flows/s", 63569, ce["xgb_native"]["throughput_per_s"]); chk("VII LLM flows/s", 3.6, 1000 / l8["edge_iiotset"]["latency_ms"]["mean"])
print("Table VIII — generalization")
g8 = ds["generalization"]
for dsn, vals in (("edge_iiotset", (1.000, 1.000, 1.000)), ("ciciot2023", (0.869, 0.850, 0.863))):
    for m, v in zip(("rf", "xgb", "llm"), vals): chk(f"VIII {dsn} {m} mean recall", v, g8[dsn][m]["mean_recall"])
for m, v in zip(("rf", "xgb", "llm"), (0.155, 0.148, 0.226)): chk(f"VIII CICIoT {m} benign FPR", v, g8["ciciot2023"][m]["mean_benign_fpr"])
for fam, vals in (("gafgyt", (0.963, 0.355, 0.982)), ("mirai", (0.999, 1.000, 0.620))):
    for m, v in zip(("rf", "xgb", "llm"), vals): chk(f"VIII N-BaIoT {fam} {m}", v, g8["nbaiot"][m]["per_fold"][fam])
print("Table IX — random-noise sensitivity (ε = 0.1) and evasion")
ae = J("v4", "adversarial_edge_iiotset_Qwen2.5-1.5B_s42_noid.json"); ac = J("v4", "adversarial_ciciot2023_Qwen2.5-1.5B_s42.json")
chk("IX Edge RF", 0.482, ae["noise_curve"]["rf_macro_f1"][1]); chk("IX Edge XGB", 0.386, ae["noise_curve"]["xgb_macro_f1"][1]); chk("IX Edge LLM", 0.037, ae["noise_curve"]["llm_macro_f1"][1])
chk("IX CICIoT RF", 0.373, ac["noise_curve"]["rf_macro_f1"][1]); chk("IX CICIoT XGB", 0.164, ac["noise_curve"]["xgb_macro_f1"][1]); chk("IX CICIoT LLM", 0.013, ac["noise_curve"]["llm_macro_f1"][1])
chk("IX CICIoT evasion LLM", 0.739, ac["evasion"]["llm_evasion_rate"])
print("Table X — attacker-constrained evaluation")
cx = J("v4", "constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")
for nm, v in (("rf", 0.471), ("xgb", 0.388), ("llm_fp16", 0.208), ("llm_int8", 0.206), ("llm_nf4", 0.209)):
    chk(f"X {nm} all-numeric ε=0.1", v, cx["noise"][nm]["all_numeric_valid"]["macro_f1"][1])
for nm, v in (("llm_fp16", 0.23), ("llm_int8", 0.17), ("llm_nf4", 0.22), ("rf", 0.00), ("xgb", 0.00)):
    chk(f"X {nm} HSJ any size", v, cx["hsj"][nm]["success_unbounded"])
for nm in ("rf", "xgb", "llm_fp16", "llm_int8", "llm_nf4"): chk(f"X {nm} HSJ ≤1σ", 0.00, cx["hsj"][nm]["success_at_budget"]["1.0"])
print(f"\n{'ALL %d CHECKS PASSED' % n if not fails else '%d of %d CHECKS FAILED: %s' % (len(fails), n, fails)}")
sys.exit(1 if fails else 0)
