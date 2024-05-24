from logging import config
from ofa.classification.elastic_nn.networks import OFAMobileNetV3Ct
from ofa.classification.run_manager.run_config import DistributedImageNetRunConfig
import horovod.torch as hvd
import torch
import torch.nn as nn
from peak_memory_efficiency import PeakMemoryEfficiency
import matplotlib.pyplot as plt
from ofa.utils.net_viz import draw_arch

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

args = {
    "path": "trained_model",
    "teacher_path": ".torch/ofa_checkpoints/0/ofa_D4_E6_K7",
    "ofa_checkpoint_path": ".torch/ofa_checkpoints/0/ofa_D4_E6_K7",
    "dynamic_batch_size": 1,
    "base_lr": 3e-2,
    "warmup_epochs": 0,
    "warmup_lr": -1,
    "ks_list": [3, 5, 7],
    "expand_list": [0.9, 1, 1.1, 1.2],
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
    "n_worker": 8,
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

num_gpus = hvd.size()

run_config = DistributedImageNetRunConfig(
    **args, num_replicas=num_gpus, rank=hvd.rank()
)

net = OFAMobileNetV3Ct(
    n_classes=run_config.data_provider.n_classes,
    bn_param=(args["bn_momentum"], args["bn_eps"]),
    dropout_rate=args["dropout"],
    base_stage_width=args["base_stage_width"],
    width_mult=args["width_mult_list"],
    ks_list=args["ks_list"],
    expand_ratio_list=args["expand_list"],
    depth_list=args["depth_list"],
)

efficiency_predictor = PeakMemoryEfficiency(ofa_net=net)

# print(net)

max_config = {
    "ks": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    "e": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "d": [4, 4, 4, 4],
    "image_size": 224,
}
#config = max_config
config = net.sample_active_subnet()
config["image_size"] = 128
print(config)

net.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])

draw_arch(
    ofa_net=net,
    resolution=config["image_size"],
    out_name=f"nets_graphs/test/subnet",
)

subnet = net.get_active_subnet()

#print(subnet)

peak_act, history = efficiency_predictor.count_peak_activation_size(
    subnet, (1, 3, config["image_size"], config["image_size"]), get_hist=True
)

# Draw histogram
plt.bar(range(len(history)), history)
plt.xlabel('Time')
plt.ylabel('Memory Occupation')
plt.title('Memory Occupation over time')
plt.savefig(f'nets_graphs/test/memory_histogram.png')