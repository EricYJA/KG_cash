import matplotlib.pyplot as plt

# Data extracted from sentiment_cache_comparison.txt
# X-axis: Similarity Thresholds (plotting as categorical strings for exact spacing)
thresholds = ['0.99', '0.95', '0.90', '0.85', '0.80']

# The different series will be Cache Size
cache_sizes = [10, 50, 100, 200, 500]

# "Gain" is the 'Diff' column (Hit Rate improvement % over exact match)
# Format: Dictionary mapping Cache Size -> list of gains corresponding to the thresholds above
webqsp_gain = {
    10:  [0.00, 0.00, 0.00, 0.04, 0.07],
    50:  [0.00, 0.14, 0.25, 0.39, 0.74],
    100: [0.07, 0.32, 0.99, 1.73, 2.37],
    200: [0.11, 0.64, 1.98, 3.08, 4.25],
    500: [0.21, 1.31, 3.68, 5.73, 7.82],
}

cwq_gain = {
    10:  [0.00, 0.04, 0.08, 0.17, 0.29],
    50:  [0.00, 0.06, 0.20, 0.41, 0.65],
    100: [0.00, 0.10, 0.30, 0.66, 1.28],
    200: [0.01, 0.19, 0.54, 1.10, 2.18],
    500: [0.02, 0.38, 1.23, 2.48, 4.50],
}

# Journal Style Configuration (single-column body text, figure printed at 100%)
plt.rcParams.update({
    "font.family": "serif",      # Matches the journal body font (Times/CM)
    "font.size": 10,             # No shrink-to-column, so match body text
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.6,
    "lines.markersize": 5
})

# A single-column journal text block is ~6.5 inches wide, so the two datasets
# stack one above the other instead of sitting side by side.
FIG_WIDTH = 6.5
PANEL_HEIGHT = 2.4
fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 2 * PANEL_HEIGHT))

# Markers and line styles for grayscale-friendly styling
markers = ['o', 's', '^', 'D', 'v']
linestyles = ['-', '--', '-.', ':', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot WebQSP
for i, size in enumerate(cache_sizes):
    axes[0].plot(thresholds, webqsp_gain[size],
                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                 label=f'Cache Size = {size}')

axes[0].set_title('WebQSP')
axes[0].set_xlabel('Similarity Threshold')
axes[0].set_ylabel('Hit Rate Gain (%)')
axes[0].grid(True, linestyle='--', alpha=0.6)


# Plot CWQ
for i, size in enumerate(cache_sizes):
    axes[1].plot(thresholds, cwq_gain[size],
                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                 label=f'Cache Size = {size}')

axes[1].set_title('Complex Web Questions (CWQ)')
axes[1].set_xlabel('Similarity Threshold')
# Stacked panels no longer share a y-axis edge, so the lower one is labelled too.
axes[1].set_ylabel('Hit Rate Gain (%)')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Single shared legend; 3 columns so the 5 entries fit the narrower text width
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)

# Adjust layout
plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save as PDF
output_path = "hit_rate_gain_vs_threshold.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot successfully saved to {output_path}")