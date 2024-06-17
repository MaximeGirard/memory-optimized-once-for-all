from ofa.classification.elastic_nn.networks import OFAMobileNetV3
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig, DistributedCIFAR10RunConfig
import horovod.torch as hvd
import torch
import torch.nn as nn
from peak_memory_efficiency import PeakMemoryEfficiency
import matplotlib.pyplot as plt
from ofa.utils.net_viz import draw_arch
from ofa.utils import MyRandomResizedCrop
import random
import numpy as np
import os
from ofa.classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.classification.elastic_nn.training.progressive_shrinking import (
    load_models,
    validate,
    train,
)

# Function to update dictionary
def update_dict(original_dict, task):
    original_dict.update(args_per_task[task])
    return original_dict

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

TEST = False

args = {
    "path": "trained_model_MIT_imagenette",
    "teacher_path": "teacher_model_MIT_imagenette/checkpoint/checkpoint.pth.tar",
    "ofa_checkpoint_path": "teacher_model_MIT_imagenette/checkpoint/checkpoint.pth.tar",
    "dynamic_batch_size": 1,
    "base_lr": 3e-2,
    "warmup_epochs": 0,
    "warmup_lr": -1,
    "ks_list": [3, 5, 7],
    "expand_list": [3, 4, 6],
    "depth_list": [2, 3, 4],
    "manual_seed": 0,
    "lr_schedule_type": "cosine",
    "base_batch_size": 64,
    "valid_size": 100,
    "opt_type": "sgd",
    "momentum": 0.9,
    "no_nesterov": False,
    "weight_decay": 3e-5,
    "label_smoothing": 0.1,
    "no_decay_keys": "bn#bias",
    "fp16_allreduce": False,
    "model_init": "he_fout",
    "validation_frequency": 1,
    "print_frequency": 10,
    "n_worker": 12,
    "resize_scale": 0.08,
    "distort_color": "tf",
    "image_size": [128, 160, 192, 224],
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

args_per_task = {
    "kernel": {
        "n_epochs": 120 if not TEST else 1,
        "base_lr": 3e-2,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [6],
        "depth_list": [4],
    },
    "depth_1": {
        "n_epochs": 25 if not TEST else 1,
        "base_lr": 2.5e-3,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [6],
        "depth_list": [3, 4],
    },
    "depth_2": {
        "n_epochs": 120 if not TEST else 1,
        "base_lr": 7.5e-3,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [6],
        "depth_list": [2, 3, 4],
    },
    "expand_1": {
        "n_epochs": 25 if not TEST else 1,
        "base_lr": 2.5e-3,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [6],
        "depth_list": [2, 3, 4],
    },
    "expand_2": {
        "n_epochs": 25 if not TEST else 1,
        "base_lr": 7.5e-3,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [4, 6],
        "depth_list": [2, 3, 4],
    },
    "expand_3": {
        "n_epochs": 25 if not TEST else 1,
        "base_lr": 7.5e-3,
        "warmup_epochs": 0,
        "warmup_lr": -1,
        "ks_list": [3, 5, 7],
        "expand_list": [3, 4, 6],
        "depth_list": [2, 3, 4],
    },
    # "expand_4": {
    #     "n_epochs": 120 if not TEST else 1,
    #     "base_lr": 7.5e-3,
    #     "warmup_epochs": 0,
    #     "warmup_lr": -1,
    #     "ks_list": [3, 5, 7],
    #     "expand_list": [1, 2, 3, 4],
    #     "depth_list": [2, 3, 4],
    # },
}

# Create directories
os.makedirs(args["path"], exist_ok=True)

# Set random seeds
torch.manual_seed(args["manual_seed"])
torch.cuda.manual_seed_all(args["manual_seed"])
np.random.seed(args["manual_seed"])
random.seed(args["manual_seed"])

# Set MyRandomResizedCrop options
MyRandomResizedCrop.CONTINUOUS = args["continuous_size"]
MyRandomResizedCrop.SYNC_DISTRIBUTED = not args["not_sync_distributed_image_size"]

# Build run config
num_gpus = hvd.size()
args["init_lr"] = args["base_lr"] * num_gpus
args["train_batch_size"] = args["base_batch_size"]
args["test_batch_size"] = args["base_batch_size"] * 4
run_config = DistributedImageNetRunConfig(
    **args, num_replicas=num_gpus, rank=hvd.rank()
)

net = OFAMobileNetV3(
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
run_manager.save_config()
run_manager.broadcast()

# Load teacher model if needed
if args["kd_ratio"] > 0:
    net.set_active_subnet(
        ks=max(args["ks_list"]),
        expand_ratio=max(args["expand_list"]),
        depth=max(args["depth_list"]),
    )
    args["teacher_model"] = net.get_active_subnet()
    args["teacher_model"].cuda()
    load_models(run_manager, args["teacher_model"], model_path=args["teacher_path"])

# Function to get validation function dictionary
def get_validation_func_dict():
    validate_func_dict = {
        "image_size_list": (
            {224} if isinstance(args["image_size"], int) else sorted({160, 224})
        ),
        "ks_list": (
            sorted(args["ks_list"])
            if task == "kernel"
            else sorted({min(args["ks_list"]), max(args["ks_list"])})
        ),
        "expand_ratio_list": sorted(
            {min(args["expand_list"]), max(args["expand_list"])}
        ),
        "depth_list": sorted({min(args["depth_list"]), max(args["depth_list"])}),
    }
    print("Validation function parameters:", validate_func_dict)
    return validate_func_dict


# Function to set network constraint
def set_net_constraint():
    dynamic_net = run_manager.net
    dynamic_net.set_constraint(args["ks_list"], constraint_type="kernel_size")
    dynamic_net.set_constraint(args["expand_list"], constraint_type="expand_ratio")
    dynamic_net.set_constraint(args["depth_list"], constraint_type="depth")
    print(
        "Net constraint set :\n ks_list=%s\n expand_ratio_list=%s\n depth_list=%s"
        % (args["ks_list"], args["expand_list"], args["depth_list"])
    )


# Train function
def train_task(task, phase=None):
    print("Task:", task)
    task_phase = task
    if phase is not None:
        print("Phase:", phase)
        task_phase += "_" + str(phase)
    args.update(args_per_task[task_phase])
    validate_func_dict = get_validation_func_dict()

    if task == "kernel":
        load_models(run_manager, run_manager.net, args["ofa_checkpoint_path"])
        set_net_constraint()
    elif task == "depth":
        args["dynamic_batch_size"] = 2
        if phase == 1:
            update_dict(args, "depth_1")
        else:
            update_dict(args, "depth_2")
        set_net_constraint()
    elif task == "expand":
        args["dynamic_batch_size"] = 4
        if phase == 1:
            update_dict(args, "expand_1")
        elif phase==2:
            update_dict(args, "expand_2")
        elif phase==3:
            update_dict(args, "expand_3")
        else:
            update_dict(args, "expand_4")
        set_net_constraint()

    train(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )


# Train tasks
tasks = ["kernel", "depth", "expand"]
depth_phases = [1, 2]
expand_phases = [1, 2, 3]

for task in tasks:
    if task == "kernel":
        train_task("kernel")
    elif task == "depth":
        for phase in depth_phases:
            train_task("depth", phase)
    elif task == "expand":
        for phase in expand_phases:
            train_task("expand", phase)

# Save the final model
run_manager.save_model()