import matplotlib.pyplot as plt
import numpy as np

# Labels for the subfigures
categories = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

# Runtime values for each method (converted from scientific notation strings to floats)
RRGD =         [2.46e+0, 2.40e+1, 6.83e+1, 3.21e+1, 4.26e+1, 2.81e+1, 6.20e+1, 5.14e+1]
eSGD =         [7.28e-1, 9.49e-1, 7.23e+1, 1.32e+0, 1.32e+0, 9.00e-1, 2.14e+0, 1.46e+0]
lowmem_eSGD =  [8.56e-1, 9.76e-1, 1.12e+2, 4.04e+0, 1.68e+0, 1.83e+0, 3.70e+0, 1.57e+0]
RR_eSGD =      [1.38e+0, 1.81e+0, 1.27e+2, 1.27e+0, 6.71e-1, 8.20e-1, 2.54e+0, 1.31e+0]

methods = ['RRGD', 'eS-GD', 'low-mem eS-GD', 'RR-eS-GD']
data = [RRGD, eSGD, lowmem_eSGD, RR_eSGD]

# Create the plot
x = np.arange(len(categories))  # positions for (a)–(h)
width = 0.2  # bar width

fig, ax = plt.subplots(figsize=(12, 6))

# Plot each method with offset for clarity
for i, (method, runtimes) in enumerate(zip(methods, data)):
    ax.bar(x + i * width, runtimes, width=width, label=method)

# Formatting
ax.set_ylabel("Runtime (s)")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(categories)
ax.set_title("Runtimes per Subfigure for Different Methods")
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

