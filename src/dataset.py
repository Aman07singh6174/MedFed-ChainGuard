import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_parkinsons(data_path: str):
    """Load the Parkinson's UPDRS CSV and split into features (X) and target (y)."""
    df = pd.read_csv(data_path)

    # Target: total_UPDRS (what we want to predict)
    # Drop non-feature columns: subject#, target columns
    target_col = "total_UPDRS"
    drop_cols = ["motor_UPDRS", "total_UPDRS"]

    y = df[target_col].values.astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)

    # Normalize features (important for neural net training)
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, y


def partition_data(X, y, num_clients: int):
    """Split the dataset into num_clients non-IID-ish partitions (one per simulated hospital)."""
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_indices = np.array_split(indices, num_clients)

    client_datasets = []
    for idx in split_indices:
        X_client = X[idx]
        y_client = y[idx]
        client_datasets.append((X_client, y_client))

    return client_datasets


def to_dataloaders(X, y, batch_size: int = 32, val_split: float = 0.2):
    """Convert numpy arrays into PyTorch train/val DataLoaders for one client."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return trainloader, valloader


def prepare_dataset(cfg):
    """Main entry point: load data, partition across clients, build dataloaders."""
    X, y = load_parkinsons(cfg.data_path)
    client_data = partition_data(X, y, cfg.num_clients)

    trainloaders = []
    valloaders = []

    for X_client, y_client in client_data:
        trainloader, valloader = to_dataloaders(X_client, y_client)
        trainloaders.append(trainloader)
        valloaders.append(valloader)

    # Use a small held-out global test set (last 10% of all data) for server-side evaluation
    test_size = int(len(X) * 0.1)
    X_test, y_test = X[-test_size:], y[-test_size:]
    testloader, _ = to_dataloaders(X_test, y_test, val_split=0.0)

    return trainloaders, valloaders, testloader
