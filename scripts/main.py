import sys
import torch
import pytorch_lightning as pl
import numpy as np
from src.models.mag_ae import MagAE, AE, PCAE
from src.utils import min_dist,is_scattered,calculate_magnitude
from tests.synthetic_data import Spheres
from torch.utils.data import TensorDataset, DataLoader, random_split

# to log output: tensorboard --logdir ./lightning_logs

def main():
    spheres = Spheres()
    data = spheres.generate(1000)

    # train_size = int(np.ceil(0.8*data.shape[0]))
    # val_size = int(data.shape[0]-np.ceil(0.8*data.shape[0]))
    # train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    data_loader = DataLoader(data,batch_size=32,shuffle=True,num_workers=8)

    model = MagAE(data.shape[1],l=0.01)
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model,data_loader)

# Check if the latent space is scattered/the magnitude difference
    val_points = 100
    val_data = spheres.generate(val_points)
    label = np.hstack([i*np.ones(val_points) for i in range(11)])

    latent = model.forward(val_data).detach()

    val_magnitude = calculate_magnitude(val_data)
    val_latent_magnitude = calculate_magnitude(latent)
    min_dist_val = min_dist(val_data).item()
    min_dist_latent = min_dist(latent).item()
    scat = is_scattered(val_data).item()
    scat_latent = is_scattered(latent).item()

    print(f'The input space is scattered {scat} with a minimal distance {min_dist_val}. It has a magnitude of {val_magnitude}.')
    print(f'The latent space is scattered {scat_latent} with a minimal distance {min_dist_latent}. It has a magnitude of {val_latent_magnitude}.')

# Test the trained model as I don't at all trust it.
    import matplotlib.pyplot as plt

    plt.scatter(latent[:,0].numpy(),latent[:,1].numpy(),c=label,cmap='twilight')
    plt.show()

if __name__=="__main__":
    main()
