import pickle
import numpy as np
import torch
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.classification.data_providers.imagenette import ImagenetDataProvider
from ofa.classification.run_manager import (
    ImagenetRunConfig,
    DistributedImageNetRunConfig,
    DistributedRunManager,
)
from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
from predictor_imagenette import Predictor
from ofa.utils.net_viz import draw_arch
from ofa.utils.peak_memory_efficiency import PeakMemoryEfficiency
import random
import matplotlib.pyplot as plt
import json
import os
import horovod.torch as hvd
from tqdm import tqdm
from ofa.utils import AverageMeter

args = {
    "path": "trained_modelV3_imagenette",
    "dataset_path": "imagenette2/",
    "res_dir": "config_infos/",
    "device": "cuda",
    "teacher_path": None,
    "dynamic_batch_size": 1,
    "base_lr": 3e-2,
    "n_epochs": 1,
    "warmup_epochs": 0,
    "warmup_lr": -1,
    "ks_list": [3, 5, 7],
    # "expand_list": [3, 4, 6],
    "expand_list": [1, 2, 3, 4],
    "depth_list": [2, 3, 4],
    "image_size": [128, 160, 192, 224],
    "manual_seed": 0,
    "lr_schedule_type": "cosine",
    "base_batch_size": 64,
    "valid_size": 100,
    "opt_type": "sgd",
    "momentum": 0.9,
    "no_nesterov": False,
    "weight_decay": 3e-5,
    "label_smoothing": 0,
    "no_decay_keys": "bn#bias",
    "fp16_allreduce": False,
    "model_init": "he_fout",
    "validation_frequency": 1,
    "print_frequency": 10,
    "n_worker": 12,
    "resize_scale": 0.08,
    "distort_color": "tf",
    "continuous_size": True,
    "not_sync_distributed_image_size": False,
    "bn_momentum": 0.1,
    "bn_eps": 1e-5,
    "dropout": 0.1,
    "base_stage_width": "proxyless",
    "width_mult_list": 1.0,
    "dy_conv_scaling_mode": 1,
    "independent_distributed_sampling": False,
    "kd_ratio": 1.0,
    "kd_type": "ce",
}

# Set default path for Imagenet data
ImagenetDataProvider.DEFAULT_PATH = args["dataset_path"]

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
# Initialize the network
net = OFAMobileNetV3CtV3(
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

# config = {
#     "ks": [7, 5, 3, 7, 5, 5, 3, 3, 5, 5, 7, 5, 3, 5, 7, 3, 5, 5, 7, 3],
#     "e": [3, 2, 2, 2, 4, 2, 1, 1, 4, 2, 3, 1, 3, 2, 3, 2, 2, 3, 1, 4],
#     "d": [3, 4, 3, 4, 2],
#     "image_size": 224,
# }
# name = "500k_constraint_V3"

# config = {
#     "ks": [7] * 20,
#     "e": [6] * 20,
#     "d": [4, 4, 4, 4, 4],
#     "image_size": 224,
# }

# name = "max_config_MIT"

# config = {
#     "ks": [3] * 20,
#     "e": [3] * 20,
#     "d": [2, 2, 2, 2, 2],
#     "image_size": 128,
# }

# config = {
#     "ks": [5, 5, 7, 5, 5, 3, 5, 3, 3, 3, 5, 7, 5, 5, 3, 5, 3, 5, 5, 7],
#     "e": [2, 2, 2, 1, 4, 2, 2, 2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
#     "d": [4, 4, 2, 4, 2],
#     "image_size": 160,
# }

# name = "under_200k_V3"

config = {
    "ks": [5, 3, 5, 3, 5, 5, 3, 5, 7, 7, 5, 3, 3, 5, 7, 5, 7, 3, 5, 3],
    "e": [4, 2, 3, 3, 4, 2, 1, 3, 4, 2, 2, 3, 2, 2, 2, 2, 1, 2, 4, 2],
    "d": [3, 3, 3, 4, 2],
    "image_size": 192,
}

name = "test"

net.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])

subnet = net.get_active_subnet()

# Evaluate the model
run_config.data_provider.assign_active_img_size(config["image_size"])
run_manager.reset_running_statistics(net)
print(net.get_current_config())

accuracies = AverageMeter()
model = run_manager.net
data_loader = run_manager.run_config.test_loader
device = "cuda" if torch.cuda.is_available() else "cpu"

subnet.eval()
with torch.no_grad():
    pbar = tqdm(total=len(data_loader), desc="Validate", position=0, leave=True)
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        accuracy = (output.argmax(1) == labels).float().mean()
        accuracies.update(accuracy.item(), images.size(0))
        # Update tqdm description with accuracy info
        pbar.set_postfix({"acc": accuracies.avg, "img_size": images.size(2)})
        pbar.update(1)
acc = accuracies.avg

draw_arch(
    ofa_net=net,
    resolution=config["image_size"],
    out_name=os.path.join(args["res_dir"], name, "subnet"),
)

peak_act, history = efficiency_predictor.count_peak_activation_size(
    subnet, (1, 3, config["image_size"], config["image_size"]), get_hist=True
)

# Draw histogram
plt.clf()
plt.bar(range(len(history)), history)
plt.xlabel("Time")
plt.ylabel("Memory Occupation")
plt.title("Memory Occupation over time")
plt.savefig(os.path.join(args["res_dir"], name, "memory_histogram.png"))
plt.show()

# log informations in a json
data = {
    "accuracy": acc,
    "peak_memory": peak_act,
    "config": config,
    "memory_history": history,
}

info_path = os.path.join(args["res_dir"], name, "info.json")
with open(info_path, "w") as f:
    json.dump(data, f)
