from omegaconf import OmegaConf
from dataset import prepare_dataset

cfg = OmegaConf.create({
    "data_path": "data/parkinsons_updrs.csv",
    "num_clients": 5
})

trainloaders, valloaders, testloader = prepare_dataset(cfg)
print("Number of clients:", len(trainloaders))
print("Batches in client 0 trainloader:", len(trainloaders[0]))
print("Test set batches:", len(testloader))
cd