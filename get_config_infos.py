import yaml
import pickle
import numpy as np
import torch
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.classification.data_providers import ImagenetDataProvider
from ofa.classification.run_manager import (
    ImagenetRunConfig,
    DistributedImageNetRunConfig,
    DistributedRunManager,
)
from ofa.utils.net_viz import draw_arch
from ofa.utils.peak_memory_efficiency import PeakMemoryEfficiency
import random
import matplotlib.pyplot as plt
import json
import os
import horovod.torch as hvd
from tqdm import tqdm
from ofa.utils import AverageMeter

# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configuration
config = load_config('config.yaml')

# Extract args and model name from config
args = config['args']

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

# Select model based on config
if args["model"] == "constant_V3":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3

    assert args["expand_list"] == [1, 2, 3, 4]
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

    assert args["expand_list"] == [1, 2, 3, 4]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3
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

efficiency_predictor = PeakMemoryEfficiency(net)

compression = hvd.Compression.fp16 if args["fp16_allreduce"] else hvd.Compression.none
run_manager = DistributedRunManager(
    args["path"],
    net,
    run_config,
    compression,
    backward_steps=args["dynamic_batch_size"],
    is_root=(hvd.rank() == 0),
)

run_manager.load_model()

# Load subnet config from YAML
subnet_config = config['subnet_config']

# Set the active subnet
net.set_active_subnet(ks=subnet_config["ks"], e=subnet_config["e"], d=subnet_config["d"])

subnet = net.get_active_subnet()

# Draw architecture
draw_arch(
    ofa_net=net,
    resolution=subnet_config["image_size"],
    out_name=os.path.join(args["res_dir"], subnet_config["name"], "subnet"),
)

peak_act, history = efficiency_predictor.count_peak_activation_size(
    subnet, (1, 3, subnet_config["image_size"], subnet_config["image_size"]), get_hist=True
)

# Draw histogram
plt.clf()
plt.bar(range(len(history)), history)
plt.xlabel("Time")
plt.ylabel("Memory Occupation")
plt.title("Memory Occupation over time")
plt.savefig(os.path.join(args["res_dir"], subnet_config["name"], "memory_histogram.png"))
plt.show()

# Log information in a json
data = {
    "peak_memory": peak_act,
    "config": subnet_config,
    "memory_history": history,
}

info_path = os.path.join(args["res_dir"], subnet_config["name"], "info.json")
with open(info_path, "w") as f:
    json.dump(data, f)