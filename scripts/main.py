import sys
import torch
import pytorch_lightning as pl
import numpy as np
from src.models.mag_ae import MagAE
from tests.synthetic_data import Spheres
from torch.utils.data import TensorDataset, DataLoader, random_split

# to log output tensorboard --logdir ./lightning_logs

def main():
    spheres = Spheres()
    data = spheres.generate()

    # train_size = int(np.ceil(0.8*data.shape[0]))
    # val_size = int(data.shape[0]-np.ceil(0.8*data.shape[0]))
    # train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    data_loader = DataLoader(data,batch_size=32,shuffle=True,num_workers=8)

    model = MagAE(101)
    trainer = pl.Trainer(max_epochs=4)
    trainer.fit(model,data_loader)

# Test the trained model as I don't at all trust it.
    import matplotlib.pyplot as plt

    label = np.hstack([i*np.ones(1000) for i in range(11)])

    low_dim = model.forward(spheres.generate_more()).detach().numpy()
    plt.scatter(low_dim[:,0],low_dim[:,1],c=label,cmap='twilight')
    plt.show()

if __name__=="__main__":
    main()
