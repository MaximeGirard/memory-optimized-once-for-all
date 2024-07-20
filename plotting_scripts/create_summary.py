import os
import json
from prettytable import PrettyTable

def get_info(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        'accuracy': data['real_accuracy'],
        'peak_memory': data['peak_memory'],
        'flops': data['flosp']  # Using sum of memory_history as a proxy for FLOPS
    }

def format_value(value, is_float=False):
    if value == 'N/A':
        return 'N/A'
    try:
        if is_float:
            return f"{float(value):.4f}"
        else:
            return f"{int(value):,}"
    except ValueError:
        return str(value)

base_dir = '../searches' 
subdirs = ['MIT_OFA', 'CompOFA', 'MC_OFA']
table = PrettyTable()
table.field_names = ["Constraint"] + subdirs

data = {subdir: {} for subdir in subdirs}

for subdir in subdirs:
    dir_path = os.path.join(base_dir, subdir)
    for item in os.listdir(dir_path):
        if item.startswith('constraint_'):
            constraint = item.split('_')[1]
            info_path = os.path.join(dir_path, item, 'info.json')
            if os.path.exists(info_path):
                data[subdir][constraint] = get_info(info_path)

all_constraints = sorted(set.union(*[set(d.keys()) for d in data.values()]))

for constraint in all_constraints:
    accuracy_row = [f"Accuracy ({constraint})"] + [format_value(data[subdir].get(constraint, {}).get('accuracy', 'N/A'), is_float=True) for subdir in subdirs]
    memory_row = [f"Peak Memory ({constraint})"] + [format_value(data[subdir].get(constraint, {}).get('peak_memory', 'N/A')) for subdir in subdirs]
    flops_row = [f"FLOPS ({constraint})"] + [format_value(data[subdir].get(constraint, {}).get('flops', 'N/A')) for subdir in subdirs]
    
    table.add_row(accuracy_row)
    table.add_row(memory_row)
    table.add_row(flops_row)

print(table)