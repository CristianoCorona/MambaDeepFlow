"""Fig. 2 of the report (img/fig_results.pdf): (a) validation rel_vy vs block resolution, (b) forest plot.
Numbers from docs/lab-notebook.md (8-seed reference reading; poster_numbers.csv). Usage: python make_fig_results.py img"""
import sys
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

out = sys.argv[1]
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, GREY, DARK = "#2a6fdb", "#e0701b", "#1aa3a3", "#9aa3ab", "#33393f"
fig, (ax1, ax) = plt.subplots(2, 1, figsize=(3.45, 3.05), gridspec_kw={"height_ratios": [1.0, 2.15]})

# (a) level curve
bm, bs = 6.067, 0.204
ax1.axhspan(bm - bs, bm + bs, color="#e3e7eb", lw=0)
ax1.axhline(bm, color=GREY, lw=0.9, ls=(0, (4, 3)))
ax1.text(3.42, bm + bs + 0.02, r"B (mean $\pm$ s.d.)", ha="right", va="bottom", color=DARK, fontsize=6.8)
lv = [(0, 5.979, 0.224), (1, 5.888, 0.096), (2, 5.731, 0.160), (3, 5.320, 0.075)]
ax1.plot([p[0] for p in lv], [p[1] for p in lv], color=BLUE, lw=1.1)
for x, m, s in lv:
    ax1.errorbar(x, m, yerr=s, fmt="o", color=BLUE, ms=3.6, lw=1.0, capsize=0)
    ax1.text(x + 0.08, m + 0.03, f"{m:.2f}", fontsize=6.3, color=DARK)
ax1.errorbar(3.22, 5.763, yerr=0.121, fmt="s", color=AQUA, ms=3.2, lw=1.0, capsize=0)
ax1.text(3.12, 5.763, r"M$^{\rm dec}_{64}$", ha="right", va="center", fontsize=6.5, color=AQUA)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels([r"M$_{8}$", r"M$_{16}$", r"M$_{32}$", r"M$_{64}$"])
ax1.set_xlim(-0.3, 3.45)
ax1.set_ylim(5.15, 6.5)
ax1.set_ylabel(r"val. rel$_{v_y}$ (%)")
ax1.text(-0.27, 6.47, "(a)", fontsize=7.5, va="top", fontweight="bold")

# (b) forest plot
rows = [(r"M$_{64}$", -0.747, -1.027, -0.495, (-1.173, -2.470, -0.431), BLUE, True),
        (r"M$^{\rm loc}_{64}$", -0.667, -0.977, -0.372, (-0.866, -1.602, -0.347), ORANGE, True),
        (r"SA$_{64}$", -0.614, -0.859, -0.390, None, DARK, False),
        (r"$-$gating", -0.746, -1.064, -0.448, None, ORANGE, False),
        (r"$-$output norm", -0.688, -0.987, -0.424, None, ORANGE, False),
        (r"$-$input norm", -0.596, -0.827, -0.394, None, ORANGE, False),
        (r"$-$expansion", -0.590, -0.881, -0.338, None, ORANGE, False),
        (r"dense conv.", -0.712, -1.053, -0.393, None, ORANGE, False),
        (r"Conv$_{64}$", -0.116, -0.316, 0.065, None, DARK, False),
        (r"M$^{\rm dec}_{64}$", -0.304, -0.529, -0.102, None, AQUA, False),
        (r"M$_{16}$", -0.179, -0.420, 0.053, None, DARK, False),
        (r"M$_{8}$", -0.09, -0.35, 0.16, None, DARK, False),
        (r"B-large", -0.001, -0.195, 0.171, None, GREY, False)]
n = len(rows)
ax.axvspan(-1.027, -0.495, color="#e3e7eb", lw=0, zorder=0)
ax.axvline(0, color=DARK, lw=0.7)
for i, (lab, m, lo, hi, test, col, bold) in enumerate(rows):
    y = n - 1 - i
    off = 0.17 if test else 0.0
    ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.1)
    ax.plot(m, y + off, "o", mfc="white", mec=col, mew=1.0, ms=3.6)
    if test:
        ax.plot([test[1], test[2]], [y - off] * 2, color=col, lw=1.1)
        ax.plot(test[0], y - off, "o", color=col, ms=3.6)
ax.set_yticks(range(n))
ax.set_yticklabels([r[0] for r in rows][::-1])
for t, r in zip(ax.get_yticklabels(), rows[::-1]):
    if r[6]:
        t.set_fontweight("bold")
ax.annotate("", xy=(-2.55, n - 4 + 0.35), xytext=(-2.55, n - 8 - 0.35),
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.text(-2.5, n - 6, r"M$^{\rm loc}_{64}$ minus" "\none ingredient", fontsize=6.2, va="center", color=ORANGE)
ax.set_xlim(-2.6, 0.35)
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel(r"$\Delta$rel$_{v_y}$ vs. B (points); left = better")
ax.plot([], [], "o", mfc="white", mec=DARK, ms=3.6, label="validation")
ax.plot([], [], "o", color=DARK, ms=3.6, label="test (pre-registered)")
ax.legend(loc="lower left", frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
ax.text(0.22, n - 0.45, "(b)", fontsize=7.5, va="top", ha="center", fontweight="bold")
for a in (ax1, ax):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(pad=0.2, h_pad=0.6)
fig.savefig(f"{out}/fig_results.pdf")
print("ok")
