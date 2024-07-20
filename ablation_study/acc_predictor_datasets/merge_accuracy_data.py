# Memory-constant OFA – A memory-optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import pickle
import glob
import os

def flatten_dict_list(dict_list):
    return [item for sublist in dict_list.values() for item in sublist]

def merge_pickle_files(input_pattern, output_file):
    merged_data = {
        "configs": [],
        "accuracies": [],
        "features": [],
        "efficiencies": []
    }
    
    # Get all pickle files matching the input pattern
    pickle_files = glob.glob(input_pattern)
    
    for file in pickle_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
        
        # Flatten and merge all fields
        for key in merged_data.keys():
            flattened = flatten_dict_list(data[key])
            merged_data[key].extend(flattened)
    
    # Save the merged data to a new pickle file
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
    
    print(f"Merged results saved to {output_file}")

# Usage
input_pattern = "config1/*.pkl"  # Adjust this path
output_file = "config1/dataset.pkl"  # Adjust this path

merge_pickle_files(input_pattern, output_file)