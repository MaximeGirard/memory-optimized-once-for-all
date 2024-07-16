import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ofa.nas.accuracy_predictor import AccuracyPredictor, MobileNetArchEncoder

# Assuming you have the AccuracyPredictor and MobileNetArchEncoder classes defined as in your previous code

class ArchDataset(Dataset):
    def __init__(self, features, accuracies):
        self.features = features
        self.accuracies = accuracies

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.accuracies[idx]

def load_data(data_path):
    all_configs = []
    all_accuracies = []
    all_features = []

    for file in os.listdir(data_path):
        if file.endswith('.pkl'):
            with open(os.path.join(data_path, file), 'rb') as f:
                data = pickle.load(f)
                all_configs.extend(data['configs'])
                all_accuracies.extend(data['accuracies'])
                all_features.extend(data['features'])

    return np.array(all_features), np.array(all_accuracies)

def train_predictor(arch_encoder, config, device='cuda'):
    # Load and prepare data
    features, accuracies = load_data(config['search_config']['acc_dataset_path'])
    
    # Center the accuracies
    accuracy_mean = np.mean(accuracies)
    centered_accuracies = accuracies - accuracy_mean

    X_train, X_val, y_train, y_val = train_test_split(features, centered_accuracies, test_size=0.2, random_state=config['args']['manual_seed'])

    train_dataset = ArchDataset(X_train, y_train)
    val_dataset = ArchDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config['args']['base_batch_size'], shuffle=True, num_workers=config['args']['n_worker'])
    val_loader = DataLoader(val_dataset, batch_size=config['args']['base_batch_size'], num_workers=config['args']['n_worker'])

    # Initialize model
    model = AccuracyPredictor(arch_encoder, device=device).to(device)

    # Set base_acc to 0 initially (since we're working with centered accuracies)
    model.base_acc.data = torch.tensor(0.0, device=device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['args']['base_lr'])

    # Training loop
    best_val_loss = float('inf')
    epochs = 100  # You might want to add this to your config file
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, accuracies in train_loader:
            features, accuracies = features.to(device), accuracies.to(device)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, accuracies)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, accuracies in val_loader:
                features, accuracies = features.to(device), accuracies.to(device)
                predictions = model(features)
                val_loss += criterion(predictions, accuracies).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['search_config']['acc_predictor_checkpoint'])

    print("Training completed.")

    # Set base_acc to the mean of the original accuracies
    model.base_acc.data = torch.tensor(accuracy_mean, device=device)

    return model

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config_search.yaml')

# Initialize arch_encoder
arch_encoder = MobileNetArchEncoder(
    image_size_list=config['args']['image_size'],
    depth_list=config['args']['depth_list'],
    expand_list=config['args']['expand_list'],
    ks_list=config['args']['ks_list'],
    n_stage=5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = train_predictor(arch_encoder, config, device=device)

print(f"Final base_acc: {trained_model.base_acc.item()}")