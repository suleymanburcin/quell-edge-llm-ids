#!/usr/bin/env python3
"""
QUELL Figure 6 - random-noise sensitivity on three datasets.
Reads every plotted value at run time from the released result files (no value is hard-coded):
  results/v4/adversarial_edge_iiotset_Qwen2.5-1.5B_s42_noid.json   (Edge-IIoTset, identifier-free; RF, XGB, LLM)
  results/v4/adversarial_ciciot2023_Qwen2.5-1.5B_s42.json          (CICIoT2023; RF, XGB, LLM)
  results/v4/adversarial_nbaiot_classic_s42.json                   (N-BaIoT; RF, XGB)
  results/adversarial_report_nbaiot_fullctx.json                     (N-BaIoT; LLM, identical protocol)
  results/v4/constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid.json (format-controlled LLM, FP16)
Writes QUELL_Fig6_robustness.png.   Usage: python3 make_fig6.py   Deps: matplotlib
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))
edge = J("v4", "adversarial_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")["noise_curve"]
cic = J("v4", "adversarial_ciciot2023_Qwen2.5-1.5B_s42.json")["noise_curve"]
nb_c = J("v4", "adversarial_nbaiot_classic_s42.json")["noise_curve"]
nb_l = J("adversarial_report_nbaiot_fullctx.json")["noise_curve"]
fmt = J("v4", "constrained_attack_edge_iiotset_Qwen2.5-1.5B_s42_noid.json")["noise"]["llm_fp16"]["all_numeric_valid"]
panels = [("Edge-IIoTset", edge["eps"], edge["rf_macro_f1"], edge["xgb_macro_f1"], edge["llm_macro_f1"], fmt),
          ("CICIoT2023", cic["eps"], cic["rf_macro_f1"], cic["xgb_macro_f1"], cic["llm_macro_f1"], None),
          ("N-BaIoT", nb_c["eps"], nb_c["rf_macro_f1"], nb_c["xgb_macro_f1"], nb_l["llm_macro_f1"], None)]
assert nb_c["eps"] == nb_l["eps"]

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito; validated (CVD dE >= 8.6)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True)
for ax, (name, eps, rf, xg, llm, fm) in zip(axes, panels):
    ax.plot(eps, rf, color=BLUE, marker="o", ms=9, lw=2.2, label="Random forest", zorder=4)
    ax.plot(eps, xg, color=GREEN, marker="s", ms=8, lw=2.2, label="XGBoost", zorder=4)
    ax.plot(eps, llm, color=ORANGE, marker="^", ms=9, lw=2.2, ls="--", label="LLM (Qwen2.5-1.5B)", zorder=4)
    if fm:
        ax.plot(fm["eps"], fm["macro_f1"], color=ORANGE, marker="^", ms=8, lw=1.8, ls=":", mfc="white",
                label="LLM, format-controlled", zorder=3)
    ax.set_title(name, fontsize=18, fontweight="bold", pad=6)
    ax.set_xlabel(r"Gaussian noise magnitude $\epsilon$ ($\times$ feature std)")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.02, 1.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.annotate(f"{llm[1]:.2f}", xy=(eps[1], llm[1]), xytext=(eps[1] + 0.06, llm[1] + 0.07), color="#333333", fontsize=14)
    ax.annotate(f"{rf[1]:.2f}", xy=(eps[1], rf[1]), xytext=(eps[1] + 0.06, rf[1] + 0.04), color="#333333", fontsize=14)
axes[0].set_ylabel("Detection quality (macro-F1)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig6_robustness.png"); fig.savefig(out, dpi=300, bbox_inches="tight"); print("saved", out)
