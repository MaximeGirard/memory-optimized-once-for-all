# Memory-constant OFA – A memory-optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import gc
import json
import os
import random

import horovod.torch as hvd
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ofa.classification.elastic_nn.training.progressive_shrinking import \
    load_models
from ofa.classification.run_manager.distributed_run_manager import \
    DistributedRunManager
from ofa.classification.run_manager.run_config import \
    DistributedImageNetRunConfig
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.utils import AverageMeter, PeakMemoryEfficiency
from ofa.utils.net_viz import draw_arch


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

# Evaluate the model
accuracies = AverageMeter()
model = run_manager.net
data_loader = run_manager.run_config.test_loader
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load subnet config from YAML
subnet_config = config["subnet_config"]

if subnet_config["random_sample"]:
    net.sample_active_subnet()
    run_config.data_provider.assign_active_img_size(224)
else:
    net.set_active_subnet(
        ks=subnet_config["ks"], e=subnet_config["e"], d=subnet_config["d"]
    )
    run_config.data_provider.assign_active_img_size(subnet_config["image_size"])
run_manager.reset_running_statistics(net)

model.eval()

with torch.no_grad():
    pbar = tqdm(total=len(data_loader), desc="Validate", position=0, leave=True)
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        accuracy = (output.argmax(1) == labels).float().mean()
        accuracies.update(accuracy.item(), images.size(0))

        pbar.set_postfix({"acc": accuracies.avg, "img_size": images.size(2)})
        pbar.update(1)

print(f"Final accuracy: {accuracies.avg}")

if subnet_config["draw_graphs"]:
    subnet = net.get_active_subnet()

    name = subnet_config["name"]

    draw_arch(
        ofa_net=net,
        resolution=subnet_config["image_size"],
        out_name=os.path.join(subnet_config["res_dir"], name, "subnet"),
    )

    efficiency_predictor = PeakMemoryEfficiency(ofa_net=net)
    peak_act, history = efficiency_predictor.count_peak_activation_size(
        subnet,
        (1, 3, subnet_config["image_size"], subnet_config["image_size"]),
        get_hist=True,
    )
    print(f"Peak memory: {peak_act}")

    flops_model = Mbv3FLOPsModel(net)
    flops = flops_model.get_efficiency(subnet_config)
    print(f"FLOPs: {flops}")  # M FLOPS

    plt.clf()
    plt.bar(range(1, len(history) + 1), history)

    # Create custom x-ticks
    x_ticks = [1] + list(range(10, len(history) + 1, 10))
    x_labels = [str(x) for x in x_ticks]

    plt.xticks(x_ticks, x_labels)

    plt.xlabel("Layer")
    plt.ylabel("Memory usage (in number of items in RAM)")
    plt.title("")
    plt.savefig(
        os.path.join(subnet_config["res_dir"], name, "memory_histogram.pdf"),
    )
    plt.show()
    
    data = {
        "accuracy": accuracies.avg,
        "peak_memory": peak_act,
        "config": config,
        "memory_history": history,
        "flops": flops,
    }

    info_path = os.path.join(subnet_config["res_dir"], name, "info.json")
    with open(info_path, "w") as f:
        json.dump(data, f)