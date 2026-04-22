import matplotlib.pyplot as plt
import numpy as np

# ── Data from updates.md ──────────────────────────────────────────────────────
top_k_labels = ["Top-10", "Top-50", "Top-100"]
top_k_webqsp = [7.22,  20.29, 28.72]
top_k_cwq    = [9.52,  19.08, 25.69]

# ── Palette ───────────────────────────────────────────────────────────────────
C_WEBQSP = "#4C72B0"
C_CWQ    = "#DD8452"

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

k_pos = [10, 50, 100]

ax.plot(k_pos, top_k_webqsp, "o-", color=C_WEBQSP,
        linewidth=2.5, markersize=9, markeredgecolor="white",
        markeredgewidth=1.5, label="WebQSP", zorder=3)
ax.plot(k_pos, top_k_cwq, "s-", color=C_CWQ,
        linewidth=2.5, markersize=9, markeredgecolor="white",
        markeredgewidth=1.5, label="CWQ", zorder=3)

for k, w, c in zip(k_pos, top_k_webqsp, top_k_cwq):
    if abs(w - c) < 3:
        # values too close — put higher label above, lower label below
        w_va, c_va = ("bottom", "top") if w > c else ("top", "bottom")
        w_off, c_off = (1.2, -1.2) if w > c else (-1.2, 1.2)
        ax.text(k, w + w_off, f"{w:.2f}%", ha="center", va=w_va,
                fontsize=9.5, color=C_WEBQSP, fontweight="bold")
        ax.text(k, c + c_off, f"{c:.2f}%", ha="center", va=c_va,
                fontsize=9.5, color=C_CWQ, fontweight="bold")
    else:
        ax.text(k, w + 1.2, f"{w:.2f}%", ha="center", va="bottom",
                fontsize=9.5, color=C_WEBQSP, fontweight="bold")
        ax.text(k, c - 1.8, f"{c:.2f}%", ha="center", va="top",
                fontsize=9.5, color=C_CWQ,    fontweight="bold")

ax.fill_between(k_pos, top_k_webqsp, alpha=0.12, color=C_WEBQSP)
ax.fill_between(k_pos, top_k_cwq,    alpha=0.12, color=C_CWQ)

ax.set_xticks(k_pos)
ax.set_xticklabels(top_k_labels, fontsize=12)
ax.set_ylabel("Coverage (% of all starting entities)", fontsize=11)
ax.set_title("Top-K Entity Coverage", fontsize=13, fontweight="bold", pad=14)
ax.set_ylim(0, 35)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, loc="upper left", framealpha=0.85, edgecolor="#cccccc")

fig.tight_layout(pad=2.5)
fig.savefig("entity_topk_coverage.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved: entity_topk_coverage.png")
