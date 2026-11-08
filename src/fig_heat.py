"""Figure 1 (greyscale, 180 mm): (a) national same-month, next-month and net effects of a +1 C temperature anomaly
and a +100 mm rainfall anomaly on monthly road deaths (Bayesian hierarchical model, posterior median and 95% CrI);
(b) change in annual road deaths under a sustained uniform warming of +1 and +2 C, naive same-month model vs
displacement-aware model. Writes outputs/Figure1.tif (600 dpi greyscale) and outputs/fig_heat.png.
"""
import pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

P = pathlib.Path(__file__).resolve().parents[1]
z = np.load(P / "outputs/bayes_hier_draws.npz", allow_pickle=True)   # own output (string array of state names)
MB = z["mu_beta"]
pc = lambda v: np.percentile(100 * np.expm1(v), [50, 2.5, 97.5])
K, DG, MG = "#000000", "#4d4d4d", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.09, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})

labels = ["Same month", "Next month", "Net (0–1)"]
for g, (name, cols, mk, mfc) in enumerate((("Temperature, +1 °C", (0, 1), "o", K), ("Rainfall, +100 mm", (2, 3), "s", "white"))):
    est = [pc(MB[:, cols[0]]), pc(MB[:, cols[1]]), pc(MB[:, cols[0]] + MB[:, cols[1]])]
    x = np.arange(3) + (g - 0.5) * 0.28
    for xi, (m, l, h) in zip(x, est):
        a.plot([xi, xi], [l, h], color=K if g == 0 else DG, lw=1.2, ls="-" if g == 0 else (0, (3, 1.5)))
    a.plot(x, [e[0] for e in est], ls="none", marker=mk, ms=5, mfc=mfc, mec=K, mew=0.9, label=name)
a.axhline(0, color=K, lw=0.6)
a.set_xticks(range(3)); a.set_xticklabels(labels)
a.set_ylabel("Change in monthly road deaths (%)")
a.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
a.text(-0.16, 1.13, "a", transform=a.transAxes, fontsize=9, fontweight="bold")

for i, dT in enumerate((1.0, 2.0)):
    for j, (lab, v, mfc) in enumerate((("Naive (same month only)", MB[:, 0] * dT, K), ("Displacement-aware (net)", (MB[:, 0] + MB[:, 1]) * dT, "white"))):
        m, l, h = pc(v); xi = i + (j - 0.5) * 0.28
        b.plot([xi, xi], [l, h], color=K if j == 0 else DG, lw=1.2, ls="-" if j == 0 else (0, (3, 1.5)))
        b.plot([xi], [m], ls="none", marker="D", ms=5, mfc=mfc, mec=K, mew=0.9, label=lab if i == 0 else None)
b.axhline(0, color=K, lw=0.6)
b.set_xticks([0, 1]); b.set_xticklabels(["+1 °C sustained", "+2 °C sustained"])
b.set_ylabel("Change in annual road deaths (%)")
b.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
b.text(-0.2, 1.13, "b", transform=b.transAxes, fontsize=9, fontweight="bold")
for ax in (a, b):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.5)
fig.savefig(P / "outputs/fig_heat.png", dpi=600)
Image.open(P / "outputs/fig_heat.png").convert("L").save(P / "outputs/Figure1.tif", dpi=(600, 600), compression="tiff_lzw")
Image.open(P / "outputs/fig_heat.png").convert("L").resize((1064, 435)).save(P / "outputs/_preview.png")
print("saved")
