#!/usr/bin/env python3
"""
QUELL Figure 6 - adversarial robustness on two datasets (Edge-IIoTset, N-BaIoT).
Reproducible: reads QUELL_Fig6_data.json, writes QUELL_Fig6_robustness.png.

Usage:  python3 make_fig6.py      Deps: matplotlib

Rationale: macro-F1 vs. black-box perturbation magnitude eps, one panel per dataset.
The random forest (blue, circles) degrades gracefully; the feature-to-text LLM (orange,
triangles) collapses. Two datasets show the ordering (RF > LLM) is a property of the
model, not of a single dataset. Black-box evasion rates are annotated per panel.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "QUELL_Fig6_data.json"), encoding="utf-8"))
eps = data["eps"]
BLUE = "#0072B2"; ORANGE = "#D55E00"

def mt(s):
    """Render special symbols via matplotlib mathtext -> encoding-independent, never mojibake."""
    return (s.replace("ε", r"$\epsilon$").replace("×", r"$\times$")
             .replace("≈", r"$\approx$").replace("→", r"$\rightarrow$"))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 1.0,
    "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
for ax, panel in zip(axes, data["panels"]):
    ax.plot(eps, panel["rf"], color=BLUE, marker="o", ms=11, lw=3.0,
            label="Random Forest", zorder=4)
    ax.plot(eps, panel["llm"], color=ORANGE, marker="^", ms=11, lw=3.0,
            ls="--", label="LLM (Qwen2.5-1.5B)", zorder=4)
    ax.set_title(panel["name"], fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel(mt(data["x_axis"]))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.03, 1.03)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # per-panel note pointing at the LLM at eps=0.1
    ax.annotate(mt(panel["note"]), xy=(0.1, panel["llm"][1]),
                xytext=(0.30, panel["llm"][1] + 0.24),
                color=ORANGE, fontsize=13,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    # evasion box
    ev = panel["evasion"]
    ax.text(0.97, 0.74,
            mt(f"Black-box evasion (ε={ev['eps']})") + f"\nRF {ev['rf']*100:.0f}%  ·  LLM {ev['llm']*100:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#bbbbbb"))

axes[0].set_ylabel(data["y_axis"])
axes[0].legend(loc="center left", bbox_to_anchor=(0.02, 0.42), fontsize=13,
               framealpha=0.95)
fig.tight_layout()
out = os.path.join(HERE, "QUELL_Fig6_robustness.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
