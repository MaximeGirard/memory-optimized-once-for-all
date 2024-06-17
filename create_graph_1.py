import os
import json
import matplotlib.pyplot as plt

# Function to extract accuracy and peak memory from JSON files in a directory
def extract_data(base_dir):
    accuracies = []
    peak_memories = []
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            json_path = os.path.join(dir_path, 'info.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    accuracy = data.get('accuracy')
                    peak_memory = data.get('peak_memory')
                    if accuracy is not None and peak_memory is not None:
                        accuracies.append(accuracy)
                        peak_memories.append(peak_memory)
    return accuracies, peak_memories

# Directories to read data from
mit_base_dir = 'graphs_ofa/MIT'
v3_base_dir = 'graphs_ofa/V3'

# Extract data from both directories
mit_accuracies, mit_peak_memories = extract_data(mit_base_dir)
v3_accuracies, v3_peak_memories = extract_data(v3_base_dir)

# Plot accuracy vs peak_memory for both data sets
plt.figure(figsize=(10, 6))
plt.scatter(mit_peak_memories, mit_accuracies, color='blue', label='MIT')
plt.scatter(v3_peak_memories, v3_accuracies, color='red', label='V3')

plt.title('Accuracy vs Peak Memory')
plt.xlabel('Peak Memory')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.savefig('accuracy_vs_peak_memory.png')