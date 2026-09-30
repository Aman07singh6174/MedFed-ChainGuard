import torch
import torch.nn as nn


class UPDRSNet(nn.Module):
    """Simple feedforward neural network to predict total_UPDRS score from voice features."""

    def __init__(self, input_dim: int):
        super(UPDRSNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(model, trainloader, epochs: int, lr: float, momentum: float, device="cpu"):
    """Train the model locally on a client's data for a given number of epochs."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


def test(model, testloader, device="cpu"):
    """Evaluate the model on a test/validation set. Returns average loss (MSE)."""
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss
    