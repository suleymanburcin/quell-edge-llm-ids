#!/usr/bin/env python3
"""
QUELL Figure 4 - cost-accuracy trade-off on the Jetson Orin Nano Super (Edge-IIoTset, identifier-free).
Reads every plotted value at run time from the released result files (no value is hard-coded):
  results/v4/jetson/jetson_summary.json       (on-device energy per flow, board level, batch-1; 20_jetson_summary.py)
  results/v4/leakage_check_edge_iiotset.json  (RF / XGBoost macro-F1, matched protocol, 5 seeds)
  results/v4/derived_stats.json               (LLM macro-F1, matched protocol, 3 seeds)
x = energy per flow (log scale), y = macro-F1 (Table III). Writes QUELL_Fig4_cost_accuracy.png.
Usage: python3 make_fig4.py      Deps: matplotlib
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", "v4")
J = lambda *p: json.load(open(os.path.join(R, *p), encoding="utf-8"))
js = J("jetson", "jetson_summary.json"); lk = J("leakage_check_edge_iiotset.json")["results"]["noid"]; ds = J("derived_stats.json")
ce = js["classic"]["edge_iiotset"]
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito, as in Figs. 5-6 (validated)
pts = [  # label, energy mJ, macro-F1, colour, marker, filled
    ("Random forest\n(ONNX Runtime)", ce["rf_onnxruntime"]["energy_mj_batch1"], lk["rf"]["mean"], BLUE, "o", True),
    ("Random forest\n(scikit-learn)", ce["rf_sklearn"]["energy_mj_batch1"], lk["rf"]["mean"], BLUE, "o", False),
    ("XGBoost", ce["xgb_native"]["energy_mj_batch1"], lk["xgb"]["mean"], GREEN, "s", True),
    ("LLM Qwen2.5-1.5B\n(INT8, llama.cpp)", js["llm"]["q8_0"]["edge_iiotset"]["energy_mj"], ds["edge_noid_detection"]["llm_mean"], ORANGE, "^", True)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.axisbelow": True})
fig, ax = plt.subplots(figsize=(8.6, 5.6))
off = {0: (1.0, -0.0035, "center", "top"), 1: (1.0, -0.0035, "center", "top"), 2: (1.0, 0.003, "center", "bottom"), 3: (1.0, 0.003, "center", "bottom")}
for k, (lab, e, f1, c, m, filled) in enumerate(pts):
    ax.scatter(e, f1, s=260, marker=m, color=c if filled else "white", edgecolor=c, linewidth=2.4, zorder=5)
    fx, dy, ha, va = off[k]
    ax.annotate(f"{lab}\n{e:,.1f} mJ" if e < 10 else f"{lab}\n{e:,.0f} mJ", (e, f1), xytext=(e * fx, f1 + dy), ha=ha, va=va,
                fontsize=12.5, color="#222222")
e_lo, e_hi = pts[0][1], pts[3][1]
ax.annotate("", xy=(e_hi, 0.8635), xytext=(e_lo, 0.8635), arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.4))
ax.text((e_lo * e_hi) ** 0.5, 0.8645, f"≈{round(e_hi / e_lo, -3):,.0f}× energy per flow", ha="center", va="bottom", fontsize=12.5, color="#444444")
ax.set_xscale("log"); ax.set_xlim(0.1, 30000); ax.set_ylim(0.86, 0.91)
ax.set_xticks([0.1, 1, 10, 100, 1000, 10000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
ax.set_xlabel("Energy per flow on the Jetson (mJ, board level, batch-1, log scale)")
ax.set_ylabel("Macro-F1 (Edge-IIoTset, Table III)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig4_cost_accuracy.png"); fig.savefig(out, dpi=300, bbox_inches="tight"); print("saved", out)
