import flwr as fl
from src.client import HospitalClient
from src.dataset import load_parkinsons, to_dataloaders

HOSPITAL_ID = 3
DATA_PATH = "data/hospital_3.csv"
SERVER_IP = "10.163.125.200:8080"

def main():
    X, y = load_parkinsons(DATA_PATH)
    trainloader, valloader = to_dataloaders(X, y)

    input_dim = X.shape[1]

    resource_profile = {
        "cpu": 0.8, "ram": 0.8, "battery": 0.9,
        "internet_speed": 0.7, "bandwidth": 0.7,
    }
    behavior_history = [1]

    client = HospitalClient(
        client_id=HOSPITAL_ID,
        trainloader=trainloader,
        valloader=valloader,
        input_dim=input_dim,
        resource_profile=resource_profile,
        behavior_history=behavior_history,
    )

    fl.client.start_client(
        server_address=SERVER_IP,
        client=client.to_client(),
    )

if __name__ == "__main__":
    main()