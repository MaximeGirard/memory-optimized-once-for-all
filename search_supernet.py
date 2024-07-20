# Memory-constant OFA – A memory-optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

# Careful : draw arch doesn't work with compOFA
#from ofa.utils.net_viz import draw_arch
import argparse
import json
import os
import pickle
import random

import horovod.torch as hvd
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ofa.classification.elastic_nn.networks import (CompOFAMobileNetV3,
                                                    OFAMobileNetV3,
                                                    OFAMobileNetV3CtV2,
                                                    OFAMobileNetV3CtV3)
from ofa.classification.elastic_nn.training.progressive_shrinking import \
    load_models
from ofa.classification.run_manager.distributed_run_manager import \
    DistributedRunManager
from ofa.classification.run_manager.run_config import \
    DistributedImageNetRunConfig
from ofa.nas.accuracy_predictor import AccuracyPredictor, MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.utils import AverageMeter, PeakMemoryEfficiency
from ofa.utils.net_viz import draw_arch


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate_model(model, data_loader, device):
    accuracies = AverageMeter()
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            accuracy = (output.argmax(1) == labels).float().mean()
            accuracies.update(accuracy.item(), images.size(0))
    return accuracies.avg

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

if args["model"] == "constant_V3":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3

    assert args["expand_list"] == [2, 3, 4]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3CtV3
elif args["model"] == "constant_V2":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV2

    assert args["expand_list"] == [1, 1.5, 2]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3CtV2
elif args["model"] == "constant_V1":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV1

    assert args["expand_list"] == [3, 4, 6]
    assert args["ks_list"] == [3, 5, 7]
    assert args["depth_list"] == [2, 3, 4]
    assert args["width_mult_list"] == 1.0
    model = OFAMobileNetV3CtV1
elif args["model"] == "MIT":
    from ofa.classification.elastic_nn.networks import OFAMobileNetV3

    assert args["expand_list"] == [3, 4, 6]
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

run_manager = DistributedRunManager(
    args["path"],
    net,
    run_config,
    None,
    backward_steps=args["dynamic_batch_size"],
    is_root=True,
)

# Load the trained model
load_models(run_manager, run_manager.net, args["checkpoint"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
    n_stage=5,
)

accuracy_predictor = AccuracyPredictor(
    arch_encoder,
    hidden_size=400,
    n_layers=3,
    checkpoint_path=search_config["acc_predictor_checkpoint"],
    device='cuda',
)

efficiency_predictor = PeakMemoryEfficiency(ofa_net=net)

finder = EvolutionFinder(
    accuracy_predictor=accuracy_predictor,
    efficiency_predictor=efficiency_predictor,
    population_size=20,
    max_time_budget=20,
)

constraints = np.linspace(search_config["max_constraint"], search_config["min_constraint"], search_config["N_constraint"], endpoint=True)
for constraint in constraints:
    best_valids, best_info = finder.run_evolution_search(constraint, verbose=True)

    found_config = best_info[1]
    peak_mem = int(best_info[2])
    pred_acc = best_info[0]

    net.set_active_subnet(ks=found_config["ks"], e=found_config["e"], d=found_config["d"])
    run_config.data_provider.assign_active_img_size(found_config["image_size"])
    
    subnet = net.get_active_subnet()
    print(subnet)

    name = "constraint_" + str(constraint) + "_search_" + str(random.randint(0, 1000))
    os.makedirs(os.path.join(search_config["res_dir"], name), exist_ok=True)

    # Careful : this does not work for compOFA
    if args["model"] != "CompOFA":
        draw_arch(
            ofa_net=net,
            resolution=found_config["image_size"],
            out_name=os.path.join(search_config["res_dir"], name, "subnet"),
        )

    peak_act, history = efficiency_predictor.count_peak_activation_size(
        subnet, (1, 3, found_config["image_size"], found_config["image_size"]), get_hist=True
    )
    
    flops_model = Mbv3FLOPsModel(net)
    flops = flops_model.get_efficiency(found_config)


    # Draw histogram
    plt.clf()
    plt.bar(range(len(history)), history)
    plt.xlabel("Time")
    plt.ylabel("Memory Occupation")
    plt.title("Memory Occupation over time")
    plt.savefig(os.path.join(search_config["res_dir"], name, "memory_histogram.png"))
    plt.show()
    print("Best Information:", best_info)
    
    # Compute real accuracy
    # Set active subnet
    run_manager.reset_running_statistics(net)

    # Evaluate the model
    real_accuracy = evaluate_model(net, run_manager.run_config.test_loader, device)

    # log informations in a json
    data = {
        "predicted_accuracy": pred_acc,
        "real_accuracy": real_accuracy,
        "peak_memory": peak_mem,
        "config": found_config,
        "memory_history": history,
        "flosp": flops
    }

    info_path = os.path.join(search_config["res_dir"], name, "info.json")
    with open(info_path, "w") as f:
        json.dump(data, f)