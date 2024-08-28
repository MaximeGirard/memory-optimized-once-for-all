# MOOFA – a Memory-Optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import json
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import stats

SET_MAX_CONTRAINT = False
MAX_CONSTRANT = 400000
PLOT_LINES = True

def process_folder(folder_path):
    accuracies = []
    constraints = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "info.json":
                constraint_match = re.search(r"constraint_(\d+\.\d+)", root)
                if constraint_match:
                    constraint = float(constraint_match.group(1))
                    with open(os.path.join(root, file), "r") as f:
                        data = json.load(f)
                        if not SET_MAX_CONTRAINT or constraint <= MAX_CONSTRANT:
                            accuracies.append(data["real_accuracy"])
                            constraints.append(constraint)
    return accuracies, constraints

# Process both folders
config1_acc, config1_constraints = process_folder("../ablation_study/searches/config1")
config2_acc, config2_constraints = process_folder("../ablation_study/searches/config2")
config3_acc, config3_constraints = process_folder("../ablation_study/searches/config3")

# Create the plot
plt.figure(figsize=(10, 6))

# Function to plot scatter and regression line
def plot_config(constraints, accuracies, label, color):
    plt.scatter(constraints, accuracies, label=label, alpha=0.7, color=color)
    
    if PLOT_LINES:
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(constraints, accuracies)
        line = slope * np.array([min(constraints), max(constraints)]) + intercept
        
        # Plot regression line
        plt.plot([min(constraints), max(constraints)], line, color=color, linestyle='dashed', alpha=0.6, linewidth=3)

# Plot configurations
plot_config(config1_constraints, config1_acc, "Configuration 1", "#1f77b4")
plot_config(config2_constraints, config2_acc, "Configuration 2", "#ff7f0e")
plot_config(config3_constraints, config3_acc, "Configuration 3", "#2ca02c")

plt.xlabel("Memory Constraint")
plt.ylabel("Top-1 Accuracy (%)")
plt.legend()

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# Save the plot
plt.savefig(
    f"../figures/Ablation_Study_Accuracy_vs_constraint{'_under_' + str(int(MAX_CONSTRANT/10e3)) + '_k' if SET_MAX_CONTRAINT else ''}{'_with_lines' if PLOT_LINES else ''}.pdf",
)