import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from predictor_imagenette import Predictor

# Load the pickle file
with open("imagenette_arch_accuracies_modelV3.pkl", "rb") as f:
    data = pickle.load(f)

# Extract features and accuracies
features = np.array(data["features"])
accuracies = np.array(data["accuracies"]).reshape(-1, 1)

X_train = features[: int(0.8 * len(features))]
y_train = accuracies[: int(0.8 * len(accuracies))]
X_test = features[int(0.8 * len(features)) :]
y_test = accuracies[int(0.8 * len(accuracies)) :]

n_epochs = 100

model = Predictor(input_size=144, device="cuda")
train_losses, test_losses = model.train(X_train, y_train, X_test, y_test, n_epochs=n_epochs)

model.save_model("imagenette_acc_predictor.pth")

# Create a plot of training and test losses
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, n_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, n_epochs + 1), test_losses, label="Test Loss", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss")
plt.legend()

outputs = model.forward(torch.tensor(X_test).float().to("cuda")).cpu().detach().numpy()

plt.subplot(1, 3, 3)
plt.scatter(
    outputs, y_test, alpha=0.5, c="blue", label="Predicted"
)
# x=y line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    c="red",
    label="Actual",
)
plt.xlabel("Predicted Accuracy")
plt.ylabel("Actual Accuracy")
plt.title("Actual vs. Predicted Accuracies")

# Compute the MSE between actual and predicted accuracies
mse = ((outputs - y_test) ** 2).mean()
plt.text(0.2, 0.8, f"MSE: {mse:.2e}", transform=plt.gca().transAxes)

# Fit the plot window
plt.tight_layout()

# Save the plot
plt.savefig("losses_and_accuracies.png")