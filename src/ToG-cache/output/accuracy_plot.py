data = {0: 0.598535692495424, 128: 0, 512: 0.614399023794997, 2048: 0}

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Sample data
cache_sizes = ['0', '128', '512', '2048']
accuracies = [0.598535692495424, 0.6186699206833435, 0.614399023794997, 0.62660158633313]

# IEEE Paper style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 300,
})

fig, ax = plt.subplots()

x_pos = np.arange(len(cache_sizes))

# Use the tab10 categorical color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
hatch_patterns = ['/', '\\', '|', '-']

# Plot bars with colors and black edges
bars = ax.bar(x_pos, accuracies, color=colors, edgecolor='black', width=0.6)

# Apply hatches
for i, bar in enumerate(bars):
    bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])

# Labels and ticks
ax.set_xlabel('Cache Size')
ax.set_ylabel('Accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(cache_sizes)

# Format the y-axis as percentages (xmax=1.0 means data is in 0.0-1.0 range)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

# Y-axis limit with error handling for all-zero arrays
max_acc = max(accuracies)
ax.set_ylim(0, (max_acc * 1.1) if max_acc > 0 else 1.0)

# Grid setup
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()

# Save and show
plt.savefig('cache_vs_accuracy.pdf', format='pdf', bbox_inches='tight')
plt.show()
