# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
def load_data():
    features = pd.read_csv('features.csv')
    labels = pd.read_csv('labels.csv')
    
    # Convert 'yes'/'no' labels to 1/0
    labels = labels['yes/no'].map({'yes': 1, 'no': 0}).values
    features = features.values.astype(float)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

class MOE(nn.Module):
    def __init__(self, num_experts, input_size, output_size, hidden_size=64):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_experts)
        ])
        
        self.gating_network = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        gating_values = self.gating_network(x)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # [num_experts, batch_size, output_size]
        expert_outputs = expert_outputs.permute(1, 0, 2)  # Rearrange to [batch_size, num_experts, output_size]
        output = torch.einsum('bi,bij->bj', gating_values, expert_outputs)  # Sum over experts
        return output

# Main function to train and test the MOE model
def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the MOE model
    model = MOE(num_experts=8, input_size=X_train.shape[1], output_size=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    for epoch in range(500):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

     # Save the model checkpoint
    torch.save(model.state_dict(), 'moe_model.pth')
    print("Saved PyTorch Model State to moe_model.pth")

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()