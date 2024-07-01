from ast import arg
from ofa.classification.networks import MobileNetV3Large
import horovod.torch as hvd
from ofa.classification.run_manager.run_config import (
    DistributedCIFAR10RunConfig,
    DistributedImageNetRunConfig,
)
import torch
from ofa.classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
from ofa.classification.elastic_nn.networks import OFAMobileNetV3
from ofa.classification.elastic_nn.training.progressive_shrinking import (
    load_models,
    validate,
    train,
)

args = {
    "path": "teacher_model_MIT_imagenette",
    "teacher_path": None,
    "dynamic_batch_size": 1,
    "base_lr": 3e-2,
    "n_epochs": 120,
    "warmup_epochs": 5,
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
    "label_smoothing": 0,
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

net.set_active_subnet(
    ks=max(args["ks_list"]),
    expand_ratio=max(args["expand_list"]),
    depth=max(args["depth_list"]),
)

teacher_net = net.get_active_subnet()

# Initialize DistributedRunManager
compression = hvd.Compression.fp16 if args["fp16_allreduce"] else hvd.Compression.none
run_manager = DistributedRunManager(
    args["path"],
    teacher_net,
    run_config,
    compression,
    backward_steps=args["dynamic_batch_size"],
    is_root=(hvd.rank() == 0),
)
run_manager.save_config()
run_manager.broadcast()

run_manager.load_model()

args["teacher_model"] = None

run_manager.train(args, warmup_epochs=args["warmup_epochs"])

run_manager.save_model()
