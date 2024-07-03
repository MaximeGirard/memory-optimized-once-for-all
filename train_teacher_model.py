import yaml
import horovod.torch as hvd
from ofa.classification.run_manager.run_config import (
    DistributedImageNetRunConfig,
)
import torch
from ofa.classification.run_manager.distributed_run_manager import (
    DistributedRunManager,
)
import wandb


# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load configuration
config = load_config("config_teacher.yaml")

# Extract args from config
args = config["args"]
wandb_config = config["wandb"]

# Initialize Horovod
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# Initialize wandb if enabled
if wandb_config["use_wandb"] and hvd.rank() == 0:
    wandb.init(project=wandb_config["project_name"], config=args, reinit=True)

print("Rank:", hvd.rank())

# Build run config
num_gpus = hvd.size()
print("Number of GPUs:", num_gpus)

# Print all visible GPU devices
print("Number of visible GPUs:", torch.cuda.device_count())
# Print their name
print("This process is running on :", torch.cuda.get_device_name(hvd.local_rank()))
# the others :
for i in range(torch.cuda.device_count()):
    if i != hvd.local_rank():
        print("Another GPU is available :", torch.cuda.get_device_name(i))

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

run_manager.train(
    args,
    warmup_epochs=args["warmup_epochs"],
    use_wandb=wandb_config["use_wandb"],
    wandb_tag="teacher",
)
run_manager.save_model()