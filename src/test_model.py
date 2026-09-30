from omegaconf import OmegaConf
from dataset import prepare_dataset
from model import UPDRSNet, train, test

cfg = OmegaConf.create({
    "data_path": "data/parkinsons_updrs.csv",
    "num_clients": 5
})

trainloaders, valloaders, testloader = prepare_dataset(cfg)

# Check input dimension from one batch
X_batch, y_batch = next(iter(trainloaders[0]))
input_dim = X_batch.shape[1]
print("Input dimension:", input_dim)

model = UPDRSNet(input_dim)
model = train(model, trainloaders[0], epochs=2, lr=0.01, momentum=0.9)
loss = test(model, valloaders[0])
print("Validation MSE after training client 0:", loss)
