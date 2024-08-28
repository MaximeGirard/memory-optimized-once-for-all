# MOOFA – a Memory-Optimized OFA architecture for tight memory constraints
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
from ast import arg
import os
import random
import horovod.torch as hvd
from matplotlib.pyplot import step
import numpy as np
import torch
import yaml
import wandb

from ofa.classification.elastic_nn.training.progressive_shrinking import (
    load_models,
    train,
    validate,
)
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
from ofa.utils import MyRandomResizedCrop


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_model_config(model_name, config_path="model_configs.yaml"):
    model_configs = load_yaml(config_path)
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in model configuration file")
    return model_configs[model_name]


def parse_arguments():
    parser = argparse.ArgumentParser(description="OFA Training Script")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["teacher", "kernel", "depth", "expand"],
        required=True,
        help="Step to perform: 'teacher' or progressive shrinking steps",
    )
    parser.add_argument("--phase", type=int, help="Phase for progressive shrinking")
    parser.add_argument(
        "--model_config_path",
        default="model_configs.yaml",
        help="Path to the model configuration file",
    )
    return parser.parse_args()


def setup_environment(args, config):
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    if config["wandb"]["use_wandb"] and hvd.rank() == 0:
        wandb.init(
            project=config["wandb"]["project_name"], config=config["args"], reinit=True
        )

    set_random_seed(config["args"]["manual_seed"])
    os.makedirs(config["args"]["path"], exist_ok=True)

    print_gpu_info()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_gpu_info():
    print("Rank:", hvd.rank())
    print("Number of GPUs:", hvd.size())
    print("Number of visible GPUs:", torch.cuda.device_count())
    print("This process is running on:", torch.cuda.get_device_name(hvd.local_rank()))
    for i in range(torch.cuda.device_count()):
        if i != hvd.local_rank():
            print("Another GPU is available:", torch.cuda.get_device_name(i))


def adjust_learning_rate(args, num_gpus):
    args["init_lr"] = args["base_lr"] * num_gpus
    args["train_batch_size"] = args["base_batch_size"]
    args["test_batch_size"] = args["base_batch_size"] * 4


def initialize_model(model_config, run_config, base_args):
    model_class_name = model_config["model_class"]
    if model_class_name == "OFAMobileNetV3CtV3":
        from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3

        model_class = OFAMobileNetV3CtV3
    elif model_class_name == "OFAMobileNetV3CtV2":
        from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV2

        model_class = OFAMobileNetV3CtV2
    elif model_class_name == "OFAMobileNetV3CtV1":
        from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV1

        model_class = OFAMobileNetV3CtV1
    elif model_class_name == "OFAMobileNetV3":
        from ofa.classification.elastic_nn.networks import OFAMobileNetV3

        model_class = OFAMobileNetV3
    elif model_class_name == "CompOFAMobileNetV3":
        raise NotImplementedError("Training for CompOFA is not supported. Please refer to https://github.com/gatech-sysml/CompOFA.")
    else:
        raise NotImplementedError(
            f"Model class '{model_class_name}' is not implemented"
        )

    return model_class(
        n_classes=run_config.data_provider.n_classes,
        bn_param=(base_args["bn_momentum"], base_args["bn_eps"]),
        dropout_rate=base_args["dropout"],
        base_stage_width=base_args["base_stage_width"],
        width_mult=model_config["width_mult_list"],
        ks_list=model_config["ks_list"],
        expand_ratio_list=model_config["expand_list"],
        depth_list=model_config["depth_list"],
    )


def create_run_manager(args, config, net, run_config):
    compression = (
        hvd.Compression.fp16 if config["fp16_allreduce"] else hvd.Compression.none
    )
    return DistributedRunManager(
        config["teacher_path"] if args.step == "teacher" else config["path"],
        net,
        run_config,
        compression,
        backward_steps=config["dynamic_batch_size"],
        is_root=(hvd.rank() == 0),
    )


def generate_constraints(tasks, tasks_phases, model_config):
    ks_list = sorted(model_config["ks_list"])
    expand_list = sorted(model_config["expand_list"])
    depth_list = sorted(model_config["depth_list"])

    constraints = {}

    for task in tasks:
        num_phases = tasks_phases[task]

        for phase in range(1, num_phases + 1):
            key = f"{task}_{phase}"

            if task == "kernel":
                constraints[key] = {
                    "task_ks_list": ks_list,
                    "task_expand_list": [max(expand_list)],
                    "task_depth_list": [max(depth_list)],
                }
            elif task == "depth":
                current_depth_list = depth_list[num_phases - phase :]
                constraints[key] = {
                    "task_ks_list": ks_list,
                    "task_expand_list": [max(expand_list)],
                    "task_depth_list": current_depth_list,
                }
            elif task == "expand":
                current_expand_list = expand_list[num_phases - phase :]
                constraints[key] = {
                    "task_ks_list": ks_list,
                    "task_expand_list": current_expand_list,
                    "task_depth_list": depth_list,
                }

    return constraints


def set_net_constraint(dynamic_net, args, config, model_config):
    tasks = config["tasks"]
    tasks_phases = config["tasks_phases"]
    constraints = generate_constraints(tasks, tasks_phases, model_config)

    step_phase = f"{args.step}_{args.phase}"
    if step_phase not in constraints:
        raise ValueError(
            f"Step-phase combination '{step_phase}' not found in generated constraints"
        )

    step_args = constraints[step_phase]

    dynamic_net.set_constraint(step_args["task_ks_list"], constraint_type="kernel_size")
    dynamic_net.set_constraint(
        step_args["task_expand_list"], constraint_type="expand_ratio"
    )
    dynamic_net.set_constraint(step_args["task_depth_list"], constraint_type="depth")

    return step_args


def update_args_for_task(args, config):
    base_args = config["args"].copy()
    args_per_task = config["args_per_task"]

    if args.step == "teacher":
        teacher_args = args_per_task.get("teacher", {})
        base_args.update(teacher_args)
    else:
        if not args.phase:
            raise ValueError("Phase is required for progressive shrinking steps")

        step_phase = f"{args.step}_{args.phase}"
        step_args = args_per_task.get(step_phase, args_per_task.get(args.step, {}))
        base_args.update(step_args)

    return base_args


def get_validation_func_dict(args, config, step_args, model_config):
    validate_func_dict = {
        "image_size_list": (
            {224} if isinstance(model_config["image_size"], int) else sorted({160, 224})
        ),
        "ks_list": (
            sorted(step_args["task_ks_list"])
            if args.step == "kernel"
            else sorted(
                {min(step_args["task_ks_list"]), max(step_args["task_ks_list"])}
            )
        ),
        "expand_ratio_list": sorted(
            {min(step_args["task_expand_list"]), max(step_args["task_expand_list"])}
        ),
        "depth_list": sorted(
            {min(step_args["task_depth_list"]), max(step_args["task_depth_list"])}
        ),
    }
    print("Validation function parameters:", validate_func_dict)
    return validate_func_dict


def main():
    args = parse_arguments()
    config = load_yaml(args.config)

    # Update base_args with task-specific arguments
    base_args = update_args_for_task(args, config)

    model_config = load_model_config(base_args["model"], args.model_config_path)

    setup_environment(args, config)
    adjust_learning_rate(base_args, hvd.size())

    # Now create run_config with the updated base_args
    run_config = DistributedImageNetRunConfig(
        **base_args,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        image_size=model_config["image_size"],
    )
    net = initialize_model(model_config, run_config, base_args)

    if args.step == "teacher":
        teacher_task(args, base_args, config, net, run_config, model_config)
    else:
        progressive_shrinking_task(
            args, base_args, config, net, run_config, model_config
        )


def teacher_task(args, base_args, config, net, run_config, model_config):
    net.set_active_subnet(
        ks=max(model_config["ks_list"]),
        expand_ratio=max(model_config["expand_list"]),
        depth=max(model_config["depth_list"]),
    )
    teacher_net = net.get_active_subnet()

    run_manager = create_run_manager(args, base_args, teacher_net, run_config)
    run_manager.save_config()
    run_manager.broadcast()
    run_manager.load_model()

    base_args["teacher_model"] = None

    run_manager.train(
        base_args,
        warmup_epochs=base_args["warmup_epochs"],
        use_wandb=config["wandb"]["use_wandb"],
        wandb_tag="teacher",
    )
    run_manager.save_model()


def progressive_shrinking_task(args, base_args, config, net, run_config, model_config):
    tasks = config["tasks"]
    tasks_phases = config["tasks_phases"]

    if args.step not in tasks:
        raise ValueError(f"Step '{args.step}' not found in config file")
    if args.step in tasks_phases:
        if args.phase not in range(1, tasks_phases[args.step] + 1):
            raise ValueError(
                f"Phase '{args.phase}' not found for step '{args.step}' in config file"
            )
    elif args.phase != 1:
        raise ValueError(f"Invalid phase '{args.phase}' for step '{args.step}'")

    run_manager = create_run_manager(args, base_args, net, run_config)
    run_manager.save_config()
    run_manager.broadcast()

    if base_args["kd_ratio"] > 0:
        load_teacher_model(base_args, net, run_manager, model_config)

    checkpoint_path = get_checkpoint_path(args, base_args, config)
    load_models(run_manager, run_manager.net, checkpoint_path)

    step_args = set_net_constraint(run_manager.net, args, config, model_config)
    print(
        "Net constraint set :\n ks_list=%s\n expand_ratio_list=%s\n depth_list=%s"
        % (
            step_args["task_ks_list"],
            step_args["task_expand_list"],
            step_args["task_depth_list"],
        )
    )

    validate_func_dict = get_validation_func_dict(args, config, step_args, model_config)

    train(
        run_manager,
        base_args,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, **validate_func_dict
        ),
        use_wandb=config["wandb"]["use_wandb"],
        wandb_tag=f"{args.step}_{args.phase}",
    )

    run_manager.save_model(model_name=f"checkpoint-{args.step}_{args.phase}.pth.tar")


def load_teacher_model(base_args, net, run_manager, model_config):
    net.set_active_subnet(
        ks=max(model_config["ks_list"]),
        expand_ratio=max(model_config["expand_list"]),
        depth=max(model_config["depth_list"]),
    )
    base_args["teacher_model"] = net.get_active_subnet()
    base_args["teacher_model"].cuda()
    teacher_path = os.path.join(
        base_args["teacher_path"], "checkpoint/model_best.pth.tar"
    )
    load_models(run_manager, base_args["teacher_model"], model_path=teacher_path)


def get_checkpoint_path(args, base_args, config):
    prev = {
        "depth": "kernel",
        "expand": "depth",
    }

    base_path = base_args["path"]
    if args.step == "kernel":
        return os.path.join(base_args["teacher_path"], "checkpoint/model_best.pth.tar")
    else:
        prev_phase = args.phase - 1
        prev_step_phase = (
            f"{args.step}_{prev_phase}"
            if prev_phase > 0
            else f"{prev[args.step]}_{config['tasks_phases'][prev[args.step]]}"
        )
        return os.path.join(
            base_path, "checkpoint", f"checkpoint-{prev_step_phase}.pth.tar"
        )


if __name__ == "__main__":
    main()
