import matplotlib.pyplot as plt
import numpy as np

# Data from sentiment_cache_comparison.txt
# Structure: thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]
cache_sizes = [10, 50, 100, 200, 500]
thresholds = ['0.99', '0.95', '0.90', '0.85', '0.80']

# Avg Sem Hit Entity Overlap (%) for WebQSP
webqsp_overlap = {
    '0.99': [0.00, 0.00, 100.00, 100.00, 100.00],
    '0.95': [0.00, 100.00, 100.00, 100.00, 100.00],
    '0.90': [0.00, 71.43, 87.50, 91.96, 96.63],
    '0.85': [100.00, 81.82, 90.82, 93.68, 95.99],
    '0.80': [100.00, 85.71, 91.79, 94.58, 95.25]
}

# Avg Sem Hit Entity Overlap (%) for CWQ
cwq_overlap = {
    '0.99': [100.00, 100.00, 100.00, 100.00, 86.67],
    '0.95': [100.00, 97.92, 96.91, 94.55, 94.87],
    '0.90': [95.24, 87.50, 90.45, 88.94, 86.88],
    '0.85': [86.96, 77.53, 78.60, 78.02, 77.39],
    '0.80': [70.00, 65.33, 67.00, 65.15, 67.29]
}

# IEEE Style Configuration
plt.rcParams.update({
    "font.family": "serif",      # Matches IEEE Times/Computer Modern
    "font.size": 17,             # Enlarged for readability
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 2.0,
    "lines.markersize": 7
})

# IEEE Double-column width is typically ~7.16 inches.
# Using a 7.16 x 2.8 size gives a nice wide aspect ratio for a two-chart arrangement
fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)

# Markers and line styles to ensure differentiation in grayscale/b&w printing
markers = ['o', 's', '^', 'D', 'v']
linestyles = ['-', '--', '-.', ':', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# Plot WebQSP
for i, thresh in enumerate(thresholds):
    axes[0].plot(cache_sizes, webqsp_overlap[thresh],
                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                 label=f'Threshold = {thresh}')

axes[0].set_title('WebQSP')
axes[0].set_xlabel('Cache Size')
axes[0].set_ylabel('Avg Sem Hit Entity Overlap (%)')
axes[0].set_xticks(cache_sizes)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_ylim(50,105)

# Plot CWQ
for i, thresh in enumerate(thresholds):
    axes[1].plot(cache_sizes, cwq_overlap[thresh],
                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                 label=f'Threshold = {thresh}')

axes[1].set_title('Complex Web Questions (CWQ)')
axes[1].set_xlabel('Cache Size')
axes[1].set_xticks(cache_sizes)
axes[1].grid(True, linestyle='--', alpha=0.6)

# Provide a single elegant legend for the whole figure
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False)

# Adjust layout to make room for the legend underneath and keep spacing tight
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save uniquely tailored for LaTeX/Overleaf
output_path = "entity_overlap_vs_cache_size.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot successfully saved to {output_path}")