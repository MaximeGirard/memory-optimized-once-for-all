import yaml
import torch
from ofa.classification.networks import MobileNetV3Large
import horovod.torch as hvd
from ofa.classification.run_manager.run_config import DistributedCIFAR10RunConfig, DistributedImageNetRunConfig
from ofa.classification.run_manager.distributed_run_manager import DistributedRunManager
from ofa.classification.elastic_nn.networks import OFAMobileNetV3
from ofa.classification.elastic_nn.training.progressive_shrinking import (
    load_models,
)

# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configuration
config = load_config('teacher_config.yaml')
args = config['args']

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

run_manager.load_model()

# Assuming teacher_path is defined in your config, otherwise you might need to add it
teacher_path = args.get("teacher_path", "teacher_model_MIT_imagenette/checkpoint/checkpoint.pth.tar")
load_models(run_manager, teacher_net, model_path=teacher_path)

# Evaluate the model
def evaluate_model(run_manager):
    run_manager.net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in run_manager.run_config.data_provider.test:
            images, labels = images.cuda(), labels.cuda()
            outputs = run_manager.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

evaluate_model(run_manager)