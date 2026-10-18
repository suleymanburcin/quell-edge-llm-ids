#!/usr/bin/env python3
"""
QUELL Figure 5 - unknown-attack generalization with whole attack families held out.
Reads every plotted value at run time from the released result files (no value is hard-coded):
  results/v4/generalization_ciciot2023_Qwen2.5-1.5B_s42_family.json
  results/v4/generalization_nbaiot_Qwen2.5-1.5B_s42_family.json
Writes QUELL_Fig5_generalization.png.   Usage: python3 make_fig5.py   Deps: matplotlib, numpy
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results", "v4")
groups = []
for ds, f in (("CICIoT2023", "generalization_ciciot2023_Qwen2.5-1.5B_s42_family.json"),
              ("N-BaIoT", "generalization_nbaiot_Qwen2.5-1.5B_s42_family.json")):
    folds = json.load(open(os.path.join(R, f), encoding="utf-8"))["folds"]
    for fam, v in folds.items():
        groups.append((ds, fam, v["rf"]["unseen_recall"], v["xgb"]["unseen_recall"], v["llm"]["unseen_recall"]))

BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito; validated
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
gap = 0.9; x, pos = [], 0.0
for i, g in enumerate(groups):
    if i and g[0] != groups[i - 1][0]: pos += gap
    x.append(pos); pos += 1.0
x = np.array(x); w = 0.27
fig, ax = plt.subplots(figsize=(8.0, 5.0))
for k, (col, lab) in enumerate(((BLUE, "Random forest"), (GREEN, "XGBoost"), (ORANGE, "LLM (Qwen2.5-1.5B)"))):
    vals = [g[2 + k] for g in groups]
    ax.bar(x + (k - 1) * (w + 0.02), vals, width=w, color=col, label=lab, edgecolor="white", linewidth=1.0, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups], rotation=35, ha="right")
ax.set_ylabel("Unseen-attack recall"); ax.set_ylim(0, 1.08)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for ds in ("CICIoT2023", "N-BaIoT"):
    xs = [xx for xx, g in zip(x, groups) if g[0] == ds]
    ax.text(np.mean(xs), 1.05, ds, ha="center", va="bottom", fontsize=15, fontweight="bold")
split = [xx for xx, g in zip(x, groups) if g[0] == "N-BaIoT"][0] - (gap + 1.0) / 2
ax.axvline(split, color="#bbbbbb", lw=1.0, ls="-", zorder=1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=14, handlelength=1.2, columnspacing=1.0)
fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig5_generalization.png"); fig.savefig(out, dpi=300, bbox_inches="tight"); print("saved", out)
