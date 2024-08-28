# MOOFA – a Memory-Optimized OFA architecture for tight memory constraints
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
import time

import horovod.torch as hvd
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ofa.classification.elastic_nn.training.progressive_shrinking import load_models
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.utils import AverageMeter, PeakMemoryEfficiency
from ofa.utils.net_viz import draw_arch
from ofa.nas.accuracy_predictor import AccuracyPredictor, MobileNetArchEncoder
from ofa.nas.search_algorithm import EvolutionFinder

def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def load_model_config(model_name, config_path="model_configs.yaml"):
    model_configs = load_yaml(config_path)
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in model configuration file")
    return model_configs[model_name]

def parse_arguments():
    parser = argparse.ArgumentParser(description="OFA Evaluation and Search Script")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--mode", choices=["eval", "search"], required=True, help="Mode: eval or search")
    parser.add_argument("--model", choices=["supernet", "teacher"], help="Model to evaluate (required for eval mode)")
    parser.add_argument("--model_config_path", default="model_configs.yaml", help="Path to the model configuration file")
    args = parser.parse_args()
    if args.mode == "eval" and args.model is None:
        parser.error("--model is required when mode is 'eval'")
    return args

def setup_environment(config):
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    torch.manual_seed(config["args"]["manual_seed"])
    torch.cuda.manual_seed_all(config["args"]["manual_seed"])
    np.random.seed(config["args"]["manual_seed"])
    random.seed(config["args"]["manual_seed"])

    os.makedirs(config["args"]["path"], exist_ok=True)

def adjust_learning_rate(args, num_gpus):
    args["init_lr"] = args["base_lr"] * num_gpus
    args["train_batch_size"] = args["base_batch_size"]
    args["test_batch_size"] = args["base_batch_size"] * 4

def update_args_for_task(args, config):
    base_args = config["args"].copy()
    args_per_task = config["args_per_task"]

    if args.model == "teacher":
        teacher_args = args_per_task.get("teacher", {})
        base_args.update(teacher_args)
    else:
        step_args = args_per_task.get(list(args_per_task.keys())[-1], {})
        base_args.update(step_args)

    return base_args

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
        from ofa.classification.elastic_nn.networks import CompOFAMobileNetV3

        model_class = CompOFAMobileNetV3
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
    compression = hvd.Compression.fp16 if config["fp16_allreduce"] else hvd.Compression.none
    return DistributedRunManager(
        config["teacher_path"] if args.model == "teacher" else config["path"],
        net,
        run_config,
        compression,
        backward_steps=config["dynamic_batch_size"],
        is_root=(hvd.rank() == 0),
    )

def evaluate_model(run_manager, subnet_config=None):
    accuracies = AverageMeter()
    model = run_manager.net
    data_loader = run_manager.run_config.test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if subnet_config:
        if subnet_config["random_sample"]:
            model.sample_active_subnet()
            run_manager.run_config.data_provider.assign_active_img_size(224)
        else:
            model.set_active_subnet(
                ks=subnet_config["ks"], e=subnet_config["e"], d=subnet_config["d"]
            )
            run_manager.run_config.data_provider.assign_active_img_size(
                subnet_config["image_size"]
            )
        run_manager.reset_running_statistics(model)

    model.eval()
    forward_times = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(total=len(data_loader), desc="Evaluate", position=0, leave=True)
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            output = model(images)
            end_time = time.time()

            forward_time = end_time - start_time
            forward_times.update(forward_time, images.size(0))

            accuracy = (output.argmax(1) == labels).float().mean()
            accuracies.update(accuracy.item(), images.size(0))

            pbar.set_postfix(
                {
                    "acc": accuracies.avg,
                    "img_size": images.size(2),
                    "fwd_time": forward_times.avg,
                }
            )
            pbar.update(1)

    print(f"Final accuracy: {accuracies.avg}")
    print(f"Average forward pass time: {forward_times.avg:.4f} seconds")
    return accuracies.avg, forward_times.avg

def draw_subnet_graphs(net, subnet_config, args):
    subnet = net.get_active_subnet()
    name = subnet_config["name"]

    if args["model"] != "CompOFA":
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

    x_ticks = [1] + list(range(10, len(history) + 1, 10))
    x_labels = [str(x) for x in x_ticks]

    plt.xticks(x_ticks, x_labels)
    plt.xlabel("Layer")
    plt.ylabel("Memory usage (in number of items in RAM)")
    plt.title("")
    plt.savefig(os.path.join(subnet_config["res_dir"], name, "memory_histogram.pdf"))
    plt.close()

    return peak_act, flops, history

def run_search(net, run_manager, model_config, search_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    arch_encoder = MobileNetArchEncoder(
        image_size_list=model_config["image_size"],
        depth_list=model_config["depth_list"],
        expand_list=model_config["expand_list"],
        ks_list=model_config["ks_list"],
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
        run_manager.run_config.data_provider.assign_active_img_size(found_config["image_size"])

        subnet = net.get_active_subnet()
        print(subnet)

        name = f"constraint_{constraint}_search_{random.randint(0, 1000)}"
        os.makedirs(os.path.join(search_config["res_dir"], name), exist_ok=True)

        if model_config["model_class"] != "CompOFAMobileNetV3":
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

        plt.clf()
        plt.bar(range(len(history)), history)
        plt.xlabel("Time")
        plt.ylabel("Memory Occupation")
        plt.title("Memory Occupation over time")
        plt.savefig(os.path.join(search_config["res_dir"], name, "memory_histogram.png"))
        plt.close()

        print("Best Information:", best_info)

        run_manager.reset_running_statistics(net)

        real_accuracy, avg_forward_time = evaluate_model(run_manager)

        data = {
            "predicted_accuracy": pred_acc,
            "real_accuracy": real_accuracy,
            "peak_memory": peak_mem,
            "config": found_config,
            "memory_history": history,
            "flops": flops,
            "avg_forward_time": avg_forward_time,
        }

        info_path = os.path.join(search_config["res_dir"], name, "info.json")
        with open(info_path, "w") as f:
            json.dump(data, f)

def main():
    args = parse_arguments()
    config = load_yaml(args.config)

    base_args = update_args_for_task(args, config)
    model_config = load_model_config(base_args["model"], args.model_config_path)

    setup_environment(config)
    adjust_learning_rate(base_args, hvd.size())

    run_config = DistributedImageNetRunConfig(
        **base_args,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        image_size=model_config["image_size"],
    )

    net = initialize_model(model_config, run_config, base_args)
    run_manager = create_run_manager(args, base_args, net, run_config)

    if args.mode == "eval":
        if args.model == "supernet":
            load_models(run_manager, run_manager.net, base_args["checkpoint"])
            subnet_config = config["subnet_config"]
            accuracy, forward_time = evaluate_model(run_manager, subnet_config)

            if subnet_config["draw_graphs"]:
                peak_memory, flops, memory_history = draw_subnet_graphs(
                    net, subnet_config, base_args
                )

                data = {
                    "accuracy": accuracy,
                    "peak_memory": peak_memory,
                    "config": config,
                    "memory_history": memory_history,
                    "flops": flops,
                    "avg_forward_time": forward_time,
                }

                info_path = os.path.join(
                    subnet_config["res_dir"], subnet_config["name"], "info.json"
                )
                with open(info_path, "w") as f:
                    json.dump(data, f)

        elif args.model == "teacher":
            net.set_active_subnet(
                ks=max(model_config["ks_list"]),
                expand_ratio=max(model_config["expand_list"]),
                depth=max(model_config["depth_list"]),
            )
            teacher_net = net.get_active_subnet()
            run_manager.net = teacher_net

            teacher_path = os.path.join(base_args["teacher_path"], "checkpoint/checkpoint.pth.tar")
            load_models(run_manager, teacher_net, model_path=teacher_path)

            accuracy, _ = evaluate_model(run_manager)

    elif args.mode == "search":
        load_models(run_manager, run_manager.net, base_args["checkpoint"])
        run_search(net, run_manager, model_config, config["search_config"])

if __name__ == "__main__":
    main()