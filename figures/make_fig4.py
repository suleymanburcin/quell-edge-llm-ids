#!/usr/bin/env python3
"""
QUELL Figure 4 - cost-accuracy trade-off (Edge-IIoTset, Jetson Orin Nano, batch-1).
Reproducible: reads QUELL_Fig4_data.json and writes QUELL_Fig4_pareto.png.

Usage:  python3 make_fig4.py
Deps:   matplotlib  (pip install matplotlib)

Design rationale (why it reads as a Pareto/cost-accuracy plot):
  * x = energy per inference on a LOG scale, because the three deployed detectors
    span an order of magnitude (140 -> 1374 mJ); a log axis keeps them legible.
  * y = macro-F1 (detection quality). "Up and to the left" = better (higher quality,
    lower energy), shown by the 'better' arrow.
  * The random forest sits top-left and therefore Pareto-dominates both LLMs:
    higher macro-F1 at ~10x lower energy. The annotated bracket makes that explicit.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "QUELL_Fig4_data.json")))
pts = data["points"]

# ---- global style: large, publication-scale fonts ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
})

fig, ax = plt.subplots(figsize=(8.6, 6.2))

for p in pts:
    ax.scatter(p["energy_mJ"], p["macro_f1"], s=340, marker=p["marker"],
               color=p["color"], edgecolor="black", linewidth=1.2, zorder=5)

# point labels (name + macro-F1), placed to avoid the markers
label_off = {
    "Random Forest":   (1.10, -0.008, "left",   "top"),
    "GPT-2-medium":    (1.00,  0.012, "center", "bottom"),
    "Qwen2.5-1.5B":    (0.92,  0.012, "right",  "bottom"),
}
for p in pts:
    fx, dy, ha, va = label_off[p["label"]]
    ax.annotate(f'{p["label"]}\n({p["detail"]}, macro-F1 {p["macro_f1"]:.3f})',
                (p["energy_mJ"], p["macro_f1"]),
                xytext=(p["energy_mJ"]*fx, p["macro_f1"]+dy),
                ha=ha, va=va, fontsize=13.5, color=p["color"], fontweight="bold")

# 'better' direction arrow (up-left)
ax.annotate("better", xy=(150, 0.983), xytext=(430, 0.965),
            fontsize=15, style="italic", color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.0))

# explicit energy-gap bracket between RF and the INT8 LLM (kept low, clear of markers)
y0 = 0.806
ax.annotate("", xy=(1374.1, y0), xytext=(140.0, y0),
            arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.6))
ax.text(118.0, y0+0.008, r"$\approx$10$\times$ energy per inference",
        ha="left", va="bottom", fontsize=12.5, color="#555555")

ax.set_xscale("log")
ax.set_xlim(100, 2000)
ax.set_ylim(0.80, 1.00)
ax.set_xticks([100, 200, 500, 1000, 2000])
ax.set_xticklabels(["100", "200", "500", "1000", "2000"])
ax.set_xlabel(data["x_axis"])
ax.set_ylabel(data["y_axis"])
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig4_pareto.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
