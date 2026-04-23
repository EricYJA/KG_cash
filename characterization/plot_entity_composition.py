import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from updates.md ──────────────────────────────────────────────────────
datasets = ["WebQSP", "CWQ"]

total  = np.array([2_893,  42_134])
unique = np.array([1_601,  10_144])
reused = total - unique

unique_pct = unique / total * 100
reused_pct = reused / total * 100

# ── Palette ───────────────────────────────────────────────────────────────────
C_UNIQUE = "#2A7F8C"
C_REUSED = "#E07B54"

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x     = np.array([0, 1])
width = 0.45

ax.bar(x, unique_pct, width,
       color=C_UNIQUE, edgecolor="white", linewidth=0.8, zorder=3)
ax.bar(x, reused_pct, width, bottom=unique_pct,
       color=C_REUSED, edgecolor="white", linewidth=0.8, zorder=3)

for i, (up, rp, u, r, t) in enumerate(zip(unique_pct, reused_pct, unique, reused, total)):
    ax.text(x[i], up / 2, f"{up:.1f}%\n({u:,})", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white")
    ax.text(x[i], up + rp / 2, f"{rp:.1f}%\n({r:,})", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white")
    ax.text(x[i], 101.5, f"Total: {t:,}", ha="center", va="bottom",
            fontsize=9, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=13, fontweight="bold")
ax.set_ylabel("Starting Entity Percentage", fontsize=11)
ax.set_title("Unique vs Reused Entities", fontsize=13, fontweight="bold", pad=30)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    mpatches.Patch(color=C_UNIQUE, label="Unique"),
    mpatches.Patch(color=C_REUSED, label="Reused"),
]
ax.legend(handles=legend_handles, fontsize=10, loc="upper center",
          bbox_to_anchor=(0.5, 1.04), ncol=2, framealpha=0.85, edgecolor="#cccccc")

fig.tight_layout(pad=2.5)
fig.savefig("entity_unique_vs_reused_mentions.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved: entity_unique_vs_reused_mentions.png")
