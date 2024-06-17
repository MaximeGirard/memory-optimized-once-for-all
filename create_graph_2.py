import os
import json
import re
import matplotlib.pyplot as plt

# Function to extract accuracy and constraint from JSON files in a directory
def extract_data(base_dir):
    accuracies = []
    constraints = []
    for dir_name in os.listdir(base_dir):
        match = re.search(r'constraint_(\d+\.?\d*)', dir_name)
        if match:
            constraint = float(match.group(1))
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.isdir(dir_path):
                json_path = os.path.join(dir_path, 'info.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as json_file:
                        data = json.load(json_file)
                        accuracy = data.get('accuracy')
                        if accuracy is not None:
                            accuracies.append(accuracy)
                            constraints.append(constraint)
    return accuracies, constraints

# Directories to read data from
mit_base_dir = 'graphs_ofa/MIT'
v3_base_dir = 'graphs_ofa/V3'

# Extract data from both directories
mit_accuracies, mit_constraints = extract_data(mit_base_dir)
v3_accuracies, v3_constraints = extract_data(v3_base_dir)

# Plot accuracy vs constraint for both data sets
plt.figure(figsize=(10, 6))
plt.scatter(mit_constraints, mit_accuracies, color='blue', label='MIT OFA')
plt.scatter(v3_constraints, v3_accuracies, color='red', label='Memory-constant OFA')

plt.title('Accuracy vs Constraint')
plt.xlabel('Constraint (Max number of parameters allowed)')
plt.ylabel('Top-1 Accuracy')
plt.legend(prop={'size': 18})
plt.grid(True)

plt.savefig('accuracy_vs_constraint.png')