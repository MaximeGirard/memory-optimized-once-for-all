import yaml
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
import horovod.torch as hvd
import torch
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

# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configuration
config = load_config('config.yaml')

# Extract args and args_per_task from config
args = config['args']
args_per_task = config['args_per_task']
tasks = config['tasks']
tasks_phases = config['tasks_phases']
TEST = config['TEST']

# Function to update dictionary
def update_dict(original_dict, task):
    original_dict.update(args_per_task[task])
    return original_dict

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

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

# Some more verification
for task in tasks:
    assert task in ['kernel', 'depth', 'expand']
    if task + "_phases" in tasks_phases.keys():
        assert len(tasks_phases[task + "_phases"]) > 0
        for phase in tasks_phases[task + "_phases"]:
            assert set(args_per_task[task + "_" + str(phase)]["ks_list"]).issubset(set(args["ks_list"]))
            assert set(args_per_task[task + "_" + str(phase)]["expand_list"]).issubset(set(args["expand_list"]))
            assert set(args_per_task[task + "_" + str(phase)]["depth_list"]).issubset(set(args["depth_list"]))

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
    teacher_path = os.path.join(args["teacher_path"], "checkpoint/model_best.pth.tar")
    load_models(run_manager, args["teacher_model"], model_path=teacher_path)

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
        pretrained_path = os.path.join(args["pretrained_model_path"], "checkpoint/model_best.pth.tar")
        load_models(run_manager, run_manager.net, pretrained_path)
        set_net_constraint()
    elif task == "depth":
        args["dynamic_batch_size"] = config['depth_dynamic_batch_size']
        if phase == 1:
            update_dict(args, "depth_1")
        else:
            update_dict(args, "depth_2")
        set_net_constraint()
    elif task == "expand":
        args["dynamic_batch_size"] = config['expand_dynamic_batch_size']
        if phase == 1:
            update_dict(args, "expand_1")
        elif phase == 2:
            update_dict(args, "expand_2")
        elif phase == 3:
            update_dict(args, "expand_3")
        set_net_constraint()

    train(
        run_manager,
        args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
    )

for task in tasks:
    if task + "_phases" in tasks_phases.keys():
        for phase in tasks_phases[task + "_phases"]:
            train_task(task, phase)
    else:
        train_task(task)

# Save the final model
run_manager.save_model()