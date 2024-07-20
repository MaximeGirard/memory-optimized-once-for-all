# Memory-constant OFA – A memory-optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import json
import matplotlib.pyplot as plt
import re

SET_MAX_CONTRAINT = False
MAX_CONSTRANT = 400000


def process_folder(folder_path):
    accuracies = []
    constraints = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "info.json":
                # Extract constraint from folder name
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
mc_ofa_accuracies, mc_ofa_constraints = process_folder("../searches/MC_OFA")
mit_ofa_accuracies, mit_ofa_constraints = process_folder("../searches/MIT_OFA")

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(
    mc_ofa_constraints, mc_ofa_accuracies, label="Memory-constant OFA", alpha=0.7
)
plt.scatter(mit_ofa_constraints, mit_ofa_accuracies, label="OFA", alpha=0.7)

plt.xlabel("Constraint (Maximum memory peak in number of items)")
plt.ylabel("Top-1 Accuracy (%)")
plt.legend()

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.show()

plt.savefig(f"../figures/Accuracy_vs_constraint{'_under_' + str(int(MAX_CONSTRANT/10e3)) + '_k' if SET_MAX_CONTRAINT else ''}.pdf")
