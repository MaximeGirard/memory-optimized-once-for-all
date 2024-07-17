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

from ofa.classification.elastic_nn.training.progressive_shrinking import load_models
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
from ofa.utils import AverageMeter, PeakMemoryEfficiency
from ofa.nas.accuracy_predictor import MobileNetArchEncoder, AccuracyPredictor
from ofa.utils.net_viz import draw_arch
from ofa.nas.search_algorithm import EvolutionFinder


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load configuration
config = load_config("config_search_MIT_OFA.yaml")

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
    population_size=10,
    max_time_budget=20,
)

constraints = np.linspace(search_config["max_constraint"], search_config["min_constraint"], search_config["N_constraint"], endpoint=False)
for constraint in constraints:
    best_valids, best_info = finder.run_evolution_search(constraint, verbose=True)

    found_config = best_info[1]
    peak_mem = int(best_info[2])
    pred_acc = best_info[0]

    net.set_active_subnet(ks=found_config["ks"], e=found_config["e"], d=found_config["d"])

    subnet = net.get_active_subnet()
    print(subnet)

    name = "constraint_" + str(constraint) + "_search_" + str(random.randint(0, 1000))

    draw_arch(
        ofa_net=net,
        resolution=found_config["image_size"],
        out_name=os.path.join(search_config["res_dir"], name, "subnet"),
    )

    peak_act, history = efficiency_predictor.count_peak_activation_size(
        subnet, (1, 3, found_config["image_size"], found_config["image_size"]), get_hist=True
    )

    # Draw histogram
    plt.clf()
    plt.bar(range(len(history)), history)
    plt.xlabel("Time")
    plt.ylabel("Memory Occupation")
    plt.title("Memory Occupation over time")
    plt.savefig(os.path.join(search_config["res_dir"], name, "memory_histogram.png"))
    plt.show()
    print("Best Information:", best_info)

    # log informations in a json
    data = {
        "predicted_accuracy": pred_acc,
        "peak_memory": peak_mem,
        "config": config,
        "memory_history": history,
    }

    info_path = os.path.join(search_config["res_dir"], name, "info.json")
    with open(info_path, "w") as f:
        json.dump(data, f)