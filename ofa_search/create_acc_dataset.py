from sympy import im
import torch
from ofa.classification.elastic_nn.networks import OFAMobileNetV3CtV3
import horovod.torch as hvd
from ofa.classification.run_manager.run_config import (
    DistributedCIFAR10RunConfig,
    DistributedImageNetRunConfig,
)
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.utils import AverageMeter
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from tqdm import tqdm
import random
import pickle

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
    "n_worker": 32,
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
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
run_manager.net.set_max_net()

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
    n_stage=5,
)

# Function to validate model
def validate_model(model, data_loader, device):
    accuracies = AverageMeter()
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
    return accuracies.avg

def test_subnet(run_manager, config):
    # print the config
    print("Testing subnet with config:")
    print(config)
    # set the active subnet
    ofa_network = run_manager.net
    ofa_network.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])
    run_config.data_provider.assign_active_img_size(config["image_size"])
    run_manager.reset_running_statistics(net=ofa_network)
    #print(ofa_network.get_current_config())
    data_loader = run_manager.run_config.test_loader
    accuracy = validate_model(ofa_network, data_loader, args["device"])
    print(f"Accuracy: {accuracy}")
    return accuracy

def test_random_subnet(run_manager, n_subnet=100):
    # randomly sample a sub-network
    configs = []
    accuracies = []
    for i in range(n_subnet):
        print(f"Testing subnet {i+1}/{n_subnet}")
        config = run_manager.net.sample_active_subnet()
        config["image_size"] = random.choice(args["image_size"])
        configs.append(config)
        acc = test_subnet(run_manager, config)
        accuracies.append(acc)
    return configs, accuracies


# Test random subnets
configs, accuracies = test_random_subnet(run_manager, n_subnet=3000)

# Map all configs to a feature
features = []
for config in configs:
    feature = arch_encoder.arch2feature(config)
    features.append(feature)
    
# save the dataset into a pickle
with open("imagenette_arch_accuracies_modelV3.pkl", "wb") as f:
    pickle.dump({
        "features": features,
        "accuracies": accuracies
    }, f)