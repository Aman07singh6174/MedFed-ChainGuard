import flwr as fl
import torch
import numpy as np
from collections import OrderedDict

from src.model import UPDRSNet, train, test


def get_parameters(model):
    """Extract model parameters as a list of numpy arrays (Flower format)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Load a list of numpy arrays into the model's state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class HospitalClient(fl.client.NumPyClient):
    """A Flower client representing one simulated hospital."""

    def __init__(self, client_id, trainloader, valloader, input_dim, resource_profile, behavior_history):
        self.client_id = client_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = UPDRSNet(input_dim)
        self.resource_profile = resource_profile      # dict: cpu, ram, battery, internet_speed, bandwidth
        self.behavior_history = behavior_history       # list of 0/1 for honest/malicious rounds

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        print(f"[Client {self.client_id}] Received params - first layer sum: {sum(p.sum() for p in parameters)}")

        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        epochs = config.get("local_epochs", 1)

        self.model = train(self.model, self.trainloader, epochs=epochs, lr=lr, momentum=momentum)

        # Return updated params, number of examples, and metrics (including resource + behavior info for trust calc)
        metrics = {
            "client_id": int(self.client_id),
            "cpu": float(self.resource_profile["cpu"]),
            "ram": float(self.resource_profile["ram"]),
            "battery": float(self.resource_profile["battery"]),
            "internet_speed": float(self.resource_profile["internet_speed"]),
            "bandwidth": float(self.resource_profile["bandwidth"]),
            "honest_round": int(self.behavior_history[-1]) if self.behavior_history else 1,
            }

        return get_parameters(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = test(self.model, self.valloader)
        return float(loss), len(self.valloader.dataset), {"mse": float(loss)}


def make_client_fn(trainloaders, valloaders, input_dim, resource_profiles, behavior_histories):
    """Factory function: returns a function Flower calls to create each client by ID."""

    def client_fn(cid: str):
        cid_int = int(cid)
        return HospitalClient(
            client_id=cid_int,
            trainloader=trainloaders[cid_int],
            valloader=valloaders[cid_int],
            input_dim=input_dim,
            resource_profile=resource_profiles[cid_int],
            behavior_history=behavior_histories[cid_int],
        ).to_client()

    return client_fn