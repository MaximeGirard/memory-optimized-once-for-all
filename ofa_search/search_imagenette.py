import pickle
import numpy as np
import torch
from ofa.nas.accuracy_predictor import MobileNetArchEncoder
from ofa.nas.efficiency_predictor import Mbv3FLOPsModel
from ofa.nas.search_algorithm import EvolutionFinder
from ofa.classification.data_providers.imagenet import ImagenetDataProvider
from ofa.classification.run_manager import ImagenetRunConfig
from ofa.classification.elastic_nn.networks import OFAMobileNetV3
from predictor_imagenette import Predictor
from ofa.utils.net_viz import draw_arch
from peak_memory_efficiency import PeakMemoryEfficiency
import random
import matplotlib.pyplot as plt
import json
import os

args = {
    "dataset_path": "imagenette2/",
    "device": "cuda",
    "res_dir": "graphs_ofa/MIT/",
    "ks_list": [3, 5, 7],
    "expand_list": [3, 4, 6],
    #"expand_list": [1, 2, 3, 4],
    "depth_list": [2, 3, 4],
    "image_size": [128, 160, 192, 224],
    "label_mapping": [0, 217, 482, 491, 497, 566, 569, 571, 574, 701],
    "remap_imagenette": True,
    "base_batch_size": 64,
    "n_workers": 12,
    "bn_momentum": 0.1,
    "bn_eps": 1e-5,
    "dropout": 0.1,
    "base_stage_width": "proxyless",
    "width_mult_list": 1.0,
}

# Set default path for Imagenet data
ImagenetDataProvider.DEFAULT_PATH = args["dataset_path"]

# Initialize Imagenet run configuration
run_config = ImagenetRunConfig(
    test_batch_size=args["base_batch_size"], n_worker=args["n_workers"]
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

# Load the pickle file
with open("imagenette_arch_accuracies_model_MIT.pkl", "rb") as f:
    data = pickle.load(f)

features = np.array(data["features"])
accuracies = np.array(data["accuracies"]).reshape(-1, 1)

X_test = torch.tensor(features[int(0.8 * len(features)) :], dtype=torch.float32)
y_test = torch.tensor(accuracies[int(0.8 * len(accuracies)) :], dtype=torch.float32)

X_test, y_test = X_test.to(args["device"]), y_test.to(args["device"])

arch_encoder = MobileNetArchEncoder(
    image_size_list=args["image_size"],
    depth_list=args["depth_list"],
    expand_list=args["expand_list"],
    ks_list=args["ks_list"],
    n_stage=5,
)

# Load the model
model = Predictor.load_model(
    "imagenette_acc_predictor_MIT.pth",
    input_size=124,
    arch_encoder=arch_encoder,
    device=args["device"],
    base_acc=0.9221,
)

#0.9221 is the base accuracy for MIT model
#0.9212 is the base accuracy for V3 model

# Evaluate the model
print("Verifying the model...")
evaluation_loss = model.evaluate(X_test, y_test)
print(f"Evaluation Loss: {evaluation_loss:.3e}")

efficiency_predictor = PeakMemoryEfficiency(ofa_net=net)

finder = EvolutionFinder(
    accuracy_predictor=model,
    efficiency_predictor=efficiency_predictor,
    population_size=10,
    max_time_budget=20,
)

N_search = 1
constraints = np.linspace(800e3, 300e3, 100, endpoint=False)
for constraint in constraints:
    # Rest of the code
        best_valids, best_info = finder.run_evolution_search(constraint, verbose=True)

        config = best_info[1]
        peak_mem = int(best_info[2])
        acc = best_info[0]

        net.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])

        subnet = net.get_active_subnet()
        print(subnet)

        name = "constraint_" + str(constraint) + "_search_" + str(N_search)

        draw_arch(
            ofa_net=net,
            resolution=config["image_size"],
            out_name=os.path.join(args["res_dir"], name, "subnet"),
        )

        peak_act, history = efficiency_predictor.count_peak_activation_size(
            subnet, (1, 3, config["image_size"], config["image_size"]), get_hist=True
        )

        # Draw histogram
        plt.clf()
        plt.bar(range(len(history)), history)
        plt.xlabel("Time")
        plt.ylabel("Memory Occupation")
        plt.title("Memory Occupation over time")
        plt.savefig(os.path.join(args["res_dir"], name, "memory_histogram.png"))
        plt.show()
        print("Best Information:", best_info)

        # log informations in a json
        data = {
            "accuracy": acc,
            "peak_memory": peak_mem,
            "config": config,
            "memory_history": history,
        }

        info_path = os.path.join(args["res_dir"], name, "info.json")
        with open(info_path, "w") as f:
            json.dump(data, f)
