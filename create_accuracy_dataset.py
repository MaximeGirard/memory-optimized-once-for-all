import argparse
import json
import os
import pickle
import random
from collections import Counter

import horovod.torch as hvd
import torch
import yaml
from tqdm import tqdm

from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
from ofa.classification.elastic_nn.training.progressive_shrinking import \
    load_models
from ofa.classification.run_manager.distributed_run_manager import \
    DistributedRunManager
from ofa.classification.run_manager.run_config import \
    DistributedImageNetRunConfig
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.utils import PeakMemoryEfficiency


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load configuration
# Argument parsing
parser = argparse.ArgumentParser(description="Memory-constant OFA")
parser.add_argument("--config", required=True, help="Path to the configuration file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Extract args from config
args = config["args"]
search_config = config["search_config"]

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# Build run config
num_gpus = hvd.size()
args["init_lr"] = args["base_lr"] * num_gpus
args["train_batch_size"] = args["base_batch_size"]
args["test_batch_size"] = args["base_batch_size"] * 4
run_config = DistributedImageNetRunConfig(
    **args, num_replicas=num_gpus, rank=hvd.rank()
)

# Model selection based on config
if args["model"] == "constant_V3":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
    assert args["expand_list"] == [2, 3, 4]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3CtV3
elif args["model"] == "constant_V2":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV2
    assert args["expand_list"] == [0.9, 1, 1.1, 1.2]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3CtV2
elif args["model"] == "MIT":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3
    assert args["expand_list"] == [3, 4, 6]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3
elif args["model"] == "CompOFA":
    from ofa.classification.elastic_nn.networks import CompOFAMobileNetV3
    assert args["expand_list"] == [3, 4, 6]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = CompOFAMobileNetV3
else:
    raise NotImplementedError

# Initialize the network
net = model(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(args["bn_momentum"], args["bn_eps"]),
    dropout_rate=args["dropout"],
    base_stage_width=args["base_stage_width"],
    width_mult=args["width_mult_list"],
    ks_list=args["ks_list"],
    expand_ratio_list=args["expand_list"],
    depth_list=args["depth_list"],
)

# Initialize DistributedRunManager
compression = hvd.Compression.fp16 if args["fp16_allreduce"] else hvd.Compression.none
run_manager = DistributedRunManager(
    args["path"],
    net,
    run_config,
    compression,
    backward_steps=args["dynamic_batch_size"],
    is_root=(hvd.rank() == 0),
)

load_models(run_manager, run_manager.net, args["checkpoint"])

# Initialize the architecture encoder
arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
    n_stage=5,
)

def create_efficiency_ranges(min_constraint, max_constraint, num_ranges):
    step = (max_constraint - min_constraint) / num_ranges
    ranges = [(min_constraint + i * step, min_constraint + (i + 1) * step) for i in range(num_ranges)]
    print(f"Efficiency ranges: {ranges}")
    return ranges

def adaptive_sample(run_manager, efficiency_predictor, min_constraint, max_constraint):
    adapted_config = run_manager.net.sample_active_subnet()
    adapted_config["image_size"] = random.choice(args["image_size"])
    adapted_config["e"] = [random.choice([2, 2, 2, 2, 2, 2, 2, 2, 3, 3]) for _ in range(20)]
    efficiency = efficiency_predictor.get_efficiency(adapted_config)
    return adapted_config, efficiency

def create_balanced_dataset(run_manager, efficiency_predictor, min_constraint, max_constraint, num_ranges, total_samples, save_interval=100, print_interval=10, max_iterations=1000000, max_iterations_fully_range=100000):
    efficiency_ranges = create_efficiency_ranges(min_constraint, max_constraint, num_ranges)
    samples_per_range = total_samples // num_ranges

    configs = {range_idx: [] for range_idx in range(num_ranges)}
    accuracies = {range_idx: [] for range_idx in range(num_ranges)}
    features = {range_idx: [] for range_idx in range(num_ranges)}
    efficiencies = {range_idx: [] for range_idx in range(num_ranges)}

    output_file = os.path.join(search_config['acc_dataset_path'], f"gpu{hvd.rank()}.pkl")

    # Load existing data if available
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            existing_data = pickle.load(f)
            for range_idx in range(num_ranges):
                configs[range_idx] = existing_data["configs"][range_idx]
                accuracies[range_idx] = existing_data["accuracies"][range_idx]
                features[range_idx] = existing_data["features"][range_idx]
                efficiencies[range_idx] = existing_data["efficiencies"][range_idx]
        print(f"GPU {hvd.rank()}: Loaded existing samples.")

    data_loader = run_manager.run_config.test_loader

    total_processed = sum(len(configs[range_idx]) for range_idx in range(num_ranges))
    pbar = tqdm(total=total_samples, initial=total_processed, desc=f"Creating dataset (GPU {hvd.rank()})")

    iterations = 0
    iterations_since_last_fill = {range_idx: 0 for range_idx in range(num_ranges)}

    while total_processed < total_samples and iterations < max_iterations:
        use_adaptive = any(iterations_since_last_fill[idx] > max_iterations_fully_range for idx in range(num_ranges))
        
        if use_adaptive:
            config, efficiency = adaptive_sample(run_manager, efficiency_predictor, min_constraint, max_constraint)
        else:
            config = run_manager.net.sample_active_subnet()
            config["image_size"] = random.choice(args["image_size"])
            efficiency = efficiency_predictor.get_efficiency(config)

        range_idx = next((i for i, (low, high) in enumerate(efficiency_ranges) if low <= efficiency < high), None)

        if range_idx is not None and len(configs[range_idx]) < samples_per_range:
            print("Found candidate config. Efficienty: ", efficiency)
            
            acc = test_subnet(run_manager, config, data_loader)
            feature = arch_encoder.arch2feature(config)

            configs[range_idx].append(config)
            accuracies[range_idx].append(acc)
            features[range_idx].append(feature)
            efficiencies[range_idx].append(efficiency)
            

            total_processed += 1
            pbar.update(1)

            iterations_since_last_fill[range_idx] = 0
            for idx in range(num_ranges):
                if idx != range_idx:
                    iterations_since_last_fill[idx] += 1

            if total_processed % print_interval == 0:
                print(f"GPU {hvd.rank()}: Processed {total_processed}/{total_samples} samples.")
                print(f"GPU {hvd.rank()}: Config: {config}, Accuracy: {acc:.2f}, Efficiency: {efficiency:.2f}")
                print(f"GPU {hvd.rank()}: Samples per range: {Counter([len(configs[i]) for i in range(num_ranges)])}")
                print(f"GPU {hvd.rank()}: Iterations since last fill: {iterations_since_last_fill}")

            if total_processed % save_interval == 0:
                print(f"GPU {hvd.rank()}: Processed {total_processed}/{total_samples} samples. Saving intermediate results...")
                save_balanced_results(configs, accuracies, features, efficiencies, output_file)
        else:
            for idx in range(num_ranges):
                iterations_since_last_fill[idx] += 1

        iterations += 1

    pbar.close()

    if iterations >= max_iterations:
        print(f"GPU {hvd.rank()}: Reached maximum iterations ({max_iterations}). Stopping.")
    else:
        print(f"GPU {hvd.rank()}: Completed dataset creation.")

    return configs, accuracies, features, efficiencies

def test_subnet(run_manager, config, subset_loader):
    ofa_network = run_manager.net
    ofa_network.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])
    run_manager.run_config.data_provider.assign_active_img_size(config["image_size"])
    run_manager.reset_running_statistics(net=ofa_network)
    accuracy = validate_model(ofa_network, subset_loader)
    return accuracy

def validate_model(model, data_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def save_balanced_results(configs, accuracies, features, efficiencies, output_file):
    dataset = {
        "configs": configs,
        "accuracies": accuracies,
        "features": features,
        "efficiencies": efficiencies
    }
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Results saved to {output_file}")

efficiency_predictor = PeakMemoryEfficiency(ofa_net=net)

# Set the constraints and parameters
MIN_CONSTRAINT = 250000  # Set this to your desired minimum value
MAX_CONSTRAINT = 800000  # Set this to your desired maximum value
NUM_RANGES = 15  # Number of efficiency ranges
TOTAL_SAMPLES = 2000  # Total number of samples in the dataset
MAX_ITERATIONS = 1000000  # Maximum number of sampling attempts
MAX_ITERATIONS_FULLY_RANGE = 1000  # Maximum iterations before triggering adaptive sampling for a range

# Create the balanced dataset
configs, accuracies, features, efficiencies = create_balanced_dataset(
    run_manager, 
    efficiency_predictor, 
    MIN_CONSTRAINT,
    MAX_CONSTRAINT,
    NUM_RANGES,
    int(TOTAL_SAMPLES / num_gpus),  # Divide total samples among GPUs
    save_interval=10, 
    print_interval=10,
    max_iterations=MAX_ITERATIONS,
    max_iterations_fully_range=MAX_ITERATIONS_FULLY_RANGE
)

# Final save
output_file = os.path.join(search_config['acc_dataset_path'], f"gpu{hvd.rank()}.pkl")
save_balanced_results(configs, accuracies, features, efficiencies, output_file)

print(f"GPU {hvd.rank()}: Balanced dataset creation completed and saved to {output_file}")