import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import flwr as fl

from src.dataset import prepare_dataset
from src.client import make_client_fn
from src.server import TrustFedAvg
from src.trust import generate_behavior_histories
from src.model import UPDRSNet


def generate_resource_profiles(num_clients, seed=42):
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(num_clients):
        if i == 1:
            profile = {
                "cpu": float(rng.uniform(0.1, 0.3)),
                "ram": float(rng.uniform(0.1, 0.3)),
                "battery": float(rng.uniform(0.1, 0.3)),
                "internet_speed": float(rng.uniform(0.1, 0.3)),
                "bandwidth": float(rng.uniform(0.1, 0.3)),
            }
        else:
            profile = {
                "cpu": float(rng.uniform(0.6, 1.0)),
                "ram": float(rng.uniform(0.6, 1.0)),
                "battery": float(rng.uniform(0.6, 1.0)),
                "internet_speed": float(rng.uniform(0.6, 1.0)),
                "bandwidth": float(rng.uniform(0.6, 1.0)),
            }
        profiles.append(profile)
    return profiles




def get_on_fit_config(cfg):
    def fit_config(server_round: int):
        return {
            "lr": cfg.config_fit.lr,
            "momentum": cfg.config_fit.momentum,
            "local_epochs": cfg.config_fit.local_epochs,
        }
    return fit_config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainloaders, valloaders, testloader = prepare_dataset(cfg)

    X_batch, _ = next(iter(trainloaders[0]))
    input_dim = X_batch.shape[1]
    print("Input dimension:", input_dim)

    resource_profiles = generate_resource_profiles(cfg.num_clients)
    behavior_histories_full = generate_behavior_histories(cfg.num_clients, cfg.num_rounds)
    behavior_histories = [[h] for h in [hist[0] for hist in behavior_histories_full]]

    client_fn = make_client_fn(trainloaders, valloaders, input_dim, resource_profiles, behavior_histories)
    init_model = UPDRSNet(input_dim)
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in init_model.state_dict().items()]
    )

    strategy = TrustFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.num_clients,
        min_evaluate_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg),
        initial_parameters=initial_parameters,
    )

    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }
    ray_init_args = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "num_cpus": 3,
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    print("\n=== Simulation complete ===")
    print(history)


if __name__ == "__main__":
    main()
   