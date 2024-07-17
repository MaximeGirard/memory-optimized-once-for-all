import json
import os
import pickle
import random

import horovod.torch as hvd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
from ofa.classification.elastic_nn.training.progressive_shrinking import load_models
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.utils import AverageMeter


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load configuration
config = load_config("config_search_CompOFA.yaml")

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


def create_dataset(run_manager, n_samples=2000, save_interval=100, print_interval=10, subset_size=5000):
    configs = []
    accuracies = []
    features = []

    # Create a unique output file for each GPU
    output_file = os.path.join(search_config['acc_dataset_path'], f"gpu{hvd.rank()}.pkl")

    # Load existing data if available
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            existing_data = pickle.load(f)
            configs = existing_data["configs"]
            accuracies = existing_data["accuracies"]
            features = existing_data["features"]
        print(f"GPU {hvd.rank()}: Loaded {len(configs)} existing samples.")

    # Get the full validation dataset
    full_val_dataset = run_manager.run_config.data_provider.valid.dataset

    # Create subset of ImageNet validation set
    subset_indices = random.sample(range(len(full_val_dataset)), k=subset_size)
    subset = Subset(full_val_dataset, subset_indices)

    subset_loader = DataLoader(
        subset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        num_workers=args["n_worker"],
        pin_memory=True,
    )

    start_index = len(configs)
    for i in tqdm(
        range(start_index, n_samples), desc=f"Creating dataset (GPU {hvd.rank()})"
    ):
        config = run_manager.net.sample_active_subnet()
        config["image_size"] = random.choice(args["image_size"])

        acc = test_subnet(run_manager, config, subset_loader)

        feature = arch_encoder.arch2feature(config)

        configs.append(config)
        accuracies.append(acc)
        features.append(feature)
        
        if (i + 1) % print_interval == 0:
            print(
                f"GPU {hvd.rank()}: Processed {i+1}/{n_samples} samples."
            )
            print(
                f"GPU {hvd.rank()}: Config: {config}, Accuracy: {acc:.2f}"
            )

        if (i + 1) % save_interval == 0:
            print(
                f"GPU {hvd.rank()}: Processed {i+1}/{n_samples} samples. Saving intermediate results..."
            )
            save_results(configs, accuracies, features, output_file)

    return configs, accuracies, features


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


def save_results(configs, accuracies, features, output_file):
    dataset = {"configs": configs, "accuracies": accuracies, "features": features}
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Results saved to {output_file}")


# Create the dataset
configs, accuracies, features = create_dataset(
    run_manager, n_samples=int(2000 / num_gpus), save_interval=100, print_interval=10, subset_size=5000
)

# Final save
output_file = os.path.join(search_config['acc_dataset_path'], f"gpu{hvd.rank()}.pkl")
save_results(configs, accuracies, features, output_file)

print(f"GPU {hvd.rank()}: Dataset creation completed and saved to {output_file}")
