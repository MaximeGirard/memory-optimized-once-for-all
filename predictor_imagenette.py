from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm


# Define the model architecture
class Predictor(nn.Module):
    def __init__(self, input_size, base_acc=None, arch_encoder=None, device="cuda"):
        super(Predictor, self).__init__()

        self.device = device
        self.base_acc = base_acc
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1, bias=False),
        )
        
        self.model = self.model.to(self.device)
        # Define loss function
        self.criterion = nn.L1Loss()

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Arch encoder if provided
        self.arch_encoder = arch_encoder

    def forward(self, x):
        if self.base_acc is None:
            raise ValueError("Base accuracy is not set")
        return self.model(x) + self.base_acc
    
    def predict_acc(self, arch_dict_list):
        if self.arch_encoder is None:
            raise ValueError("Arch encoder is not provided")
        X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        return self.forward(X)
    
    def set_base_acc(self, y_train):
        self.base_acc = y_train.mean()
        print(f"Base accuracy set to {self.base_acc}")

    
    def train(self, X_train, y_train, X_test, y_test, n_epochs=75):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.set_base_acc(y_train)
        
        y_train -= self.base_acc
        y_test -= self.base_acc
        # Create DataLoader for batching and shuffling data
        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Lists to store training and test losses
        train_losses = []
        test_losses = []

        # Train the model
        for epoch in (pbar := tqdm(range(n_epochs))):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_dataloader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Calculate test loss
            with torch.no_grad():
                self.model.eval()
                test_loss = 0.0
                for inputs, labels in test_dataloader:
                    outputs = self.model(inputs)
                    test_loss += self.criterion(outputs, labels).item()

            # Append average losses
            train_losses.append(train_loss / len(train_dataloader))
            test_losses.append(test_loss / len(test_dataloader))

            pbar.set_description(
                f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]:.3e}, Test Loss: {test_losses[-1]:.3e}"
            )
        
        return train_losses, test_losses
            
    def evaluate(self, X_test, y_test):
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        y_test -= self.base_acc
        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Calculate test loss
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            for inputs, labels in test_dataloader:
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, labels).item()
        return test_loss / len(test_dataloader)
            
    def save_model(self, filepath="imagenette_acc_predictor.pth"):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
            
    
    @staticmethod
    def load_model(model_path, input_size, arch_encoder=None, device="cuda", base_acc=0.9774):
        model = Predictor(input_size, base_acc, arch_encoder=arch_encoder, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model