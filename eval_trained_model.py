import torch
from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
import horovod.torch as hvd
from ofa.classification.run_manager.run_config import (
    DistributedCIFAR10RunConfig,
    DistributedImageNetRunConfig,
)
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from tqdm import tqdm
from ofa.utils import AverageMeter
import random

# Arguments (should match the arguments used during training)
args = {
    "path": "trained_modelV3_imagenette",
    "teacher_path": None,
    "dynamic_batch_size": 1,
    "base_lr": 3e-2,
    "n_epochs": 1,
    "warmup_epochs": 0,
    "warmup_lr": -1,
    "ks_list": [3, 5, 7],
    "expand_list": [1, 2, 3, 4],
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

run_manager.load_model()

# Evaluate the model
accuracies = AverageMeter()
model = run_manager.net
data_loader = run_manager.run_config.test_loader
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "ks": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    "e": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    "d": [4, 4, 4, 4],
    "image_size": 224,
}

net.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])
run_config.data_provider.assign_active_img_size(config["image_size"])
run_manager.reset_running_statistics(net)
print(net.get_current_config())

model.eval()
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
