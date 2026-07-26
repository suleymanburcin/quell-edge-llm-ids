#!/usr/bin/env python3
"""
QUELL Figure 5 - unknown-attack generalization (leave-one-attack-out, Edge-IIoTset).
Reproducible: reads QUELL_Fig5_data.json, writes QUELL_Fig5_generalization.png.

Usage:  python3 make_fig5.py      Deps: matplotlib

Rationale: horizontal bars = the LLM's unseen-attack recall for each held-out class
(sorted best -> worst, top -> bottom). The random forest reaches 1.000 on every fold,
drawn as the dashed reference line, so any bar short of the line is an LLM-only failure.
The worst case (DDoS_TCP, 0.50) is the honest weak point that the bar chart exposes.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "QUELL_Fig5_data.json")))
folds = data["folds"]  # already ordered best -> worst

BLUE = "#0072B2"; ORANGE = "#D55E00"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.axisbelow": True,
})

names = [f["attack"] for f in folds]
vals = [f["llm"] for f in folds]
y = list(range(len(folds)))[::-1]   # first fold at top

fig, ax = plt.subplots(figsize=(9.4, 6.6))
bars = ax.barh(y, vals, color=ORANGE, edgecolor="black", linewidth=0.8,
               height=0.66, zorder=3, label="LLM (Qwen2.5-1.5B)")
# RF reference line at 1.000
ax.axvline(data["rf_reference"], color=BLUE, linestyle="--", linewidth=2.4,
           zorder=4, label=f'Random Forest (all folds = {data["rf_reference"]:.3f})')

# value labels at bar ends
for yi, v in zip(y, vals):
    ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left",
            fontsize=12, color="#333333", fontweight="bold")

ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.0, 1.08)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel(data["x_axis"])
ax.grid(axis="x", color="#e3e3e3", linewidth=1.0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
          fontsize=13, frameon=False)
ax.set_title(f'LLM (Qwen2.5-1.5B) mean unseen recall = {data["llm_mean"]:.3f}   vs   RF = {data["rf_mean"]:.3f} (all folds)',
             fontsize=14, fontweight="bold", pad=10)

fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig5_generalization.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
