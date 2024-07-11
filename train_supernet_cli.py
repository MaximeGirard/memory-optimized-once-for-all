import yaml
import argparse
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
import horovod.torch as hvd
import torch
from ofa.utils import MyRandomResizedCrop
import random
import numpy as np
import os
import wandb
from ofa.classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.classification.elastic_nn.training.progressive_shrinking import (
    load_models,
    validate,
    train,
)


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Parse command line arguments
parser = argparse.ArgumentParser(description="OFA Training Script")
parser.add_argument(
    "--step", type=str, choices=["kernel", "depth", "expand"], required=True
)
parser.add_argument("--phase", type=int, required=True)
args = parser.parse_args()

# Load configuration
config = load_config("config_cli.yaml")

# Extract args and args_per_task from config
TEST = config["TEST"]
base_args = config["args"]
args_per_task = config["args_per_task"]
tasks = config["tasks"]
tasks_phases = config["tasks_phases"]
wandb_config = config["wandb"]

# Verify that the step and phase are in the config file
step_phase = f"{args.step}_{args.phase}"
if args.step not in tasks:
    raise ValueError(f"Step '{args.step}' not found in config file")
if args.step + "_phases" in tasks_phases:
    if args.phase not in tasks_phases[args.step + "_phases"]:
        raise ValueError(
            f"Phase '{args.phase}' not found for step '{args.step}' in config file"
        )
elif args.phase != 1:
    raise ValueError(f"Invalid phase '{args.phase}' for step '{args.step}'")

# Load specific parameters for this step and phase
step_args = (
    args_per_task[step_phase]
    if step_phase in args_per_task
    else args_per_task[args.step]
)
base_args.update(step_args)

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# Initialize wandb if enabled
if wandb_config["use_wandb"] and hvd.rank() == 0:
    wandb.init(project=wandb_config["project_name"], config=base_args, reinit=True)

# Create directories
os.makedirs(base_args["path"], exist_ok=True)

# Set random seeds
torch.manual_seed(base_args["manual_seed"])
torch.cuda.manual_seed_all(base_args["manual_seed"])
np.random.seed(base_args["manual_seed"])
random.seed(base_args["manual_seed"])

# Set MyRandomResizedCrop options
MyRandomResizedCrop.CONTINUOUS = base_args["continuous_size"]
MyRandomResizedCrop.SYNC_DISTRIBUTED = not base_args["not_sync_distributed_image_size"]

print("Rank:", hvd.rank())

# Build run config
num_gpus = hvd.size()
print("Number of GPUs:", num_gpus)
base_args["init_lr"] = base_args["base_lr"] * num_gpus
base_args["train_batch_size"] = base_args["base_batch_size"]
base_args["test_batch_size"] = base_args["base_batch_size"] * 4
run_config = DistributedImageNetRunConfig(
    **base_args, num_replicas=num_gpus, rank=hvd.rank()
)

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

net = model(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(base_args["bn_momentum"], base_args["bn_eps"]),
    dropout_rate=base_args["dropout"],
    base_stage_width=base_args["base_stage_width"],
    width_mult=base_args["width_mult_list"],
    ks_list=base_args["ks_list"],
    expand_ratio_list=base_args["expand_list"],
    depth_list=base_args["depth_list"],
)

# Initialize DistributedRunManager
compression = (
    hvd.Compression.fp16 if base_args["fp16_allreduce"] else hvd.Compression.none
)
run_manager = DistributedRunManager(
    base_args["path"],
    net,
    run_config,
    compression,
    backward_steps=base_args["dynamic_batch_size"],
    is_root=(hvd.rank() == 0),
)
run_manager.save_config()
run_manager.broadcast()

prev = {
    "depth": "kernel",
    "expand": "depth",
}

# Load checkpoint
base_path = base_args["path"]
if args.step == "kernel":
    checkpoint_path = os.path.join(
        base_args["teacher_path"], "checkpoint/model_best.pth.tar"
    )
else:
    prev_phase = args.phase - 1
    prev_step_phase = f"{args.step}_{prev_phase}" if prev_phase > 0 else prev[args.step]
    checkpoint_path = os.path.join(
        base_path, "checkpoint", f"checkpoint-{prev_step_phase}.pth.tar"
    )

load_models(run_manager, run_manager.net, checkpoint_path)


# Set network constraint
def set_net_constraint():
    dynamic_net = run_manager.net
    dynamic_net.set_constraint(base_args["task_ks_list"], constraint_type="kernel_size")
    dynamic_net.set_constraint(base_args["task_expand_list"], constraint_type="expand_ratio")
    dynamic_net.set_constraint(base_args["task_depth_list"], constraint_type="depth")
    print(
        "Net constraint set :\n ks_list=%s\n expand_ratio_list=%s\n depth_list=%s"
        % (base_args["ks_list"], base_args["expand_list"], base_args["depth_list"])
    )


set_net_constraint()


# Function to get validation function dictionary
def get_validation_func_dict():
    validate_func_dict = {
        "image_size_list": (
            {224} if isinstance(base_args["image_size"], int) else sorted({160, 224})
        ),
        "ks_list": (
            sorted(base_args["ks_list"])
            if args.step == "kernel"
            else sorted({min(base_args["ks_list"]), max(base_args["ks_list"])})
        ),
        "expand_ratio_list": sorted(
            {min(base_args["expand_list"]), max(base_args["expand_list"])}
        ),
        "depth_list": sorted(
            {min(base_args["depth_list"]), max(base_args["depth_list"])}
        ),
    }
    print("Validation function parameters:", validate_func_dict)
    return validate_func_dict


validate_func_dict = get_validation_func_dict()

# Train
train(
    run_manager,
    base_args,
    lambda _run_manager, epoch, is_test: validate(
        _run_manager, epoch, is_test, **validate_func_dict
    ),
    use_wandb=wandb_config["use_wandb"],
    wandb_tag=step_phase,
)

# Save the model
run_manager.save_model(model_name=f"checkpoint-{step_phase}.pth.tar")
