import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MagAE(pl.LightningModule):
    def __init__(self,input_size):
        super(MagAE,self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32,32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32,2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32,32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32,self.input_size)
        )
    def forward(self, x):
        h = self.encoder(x)
        return h

    def training_step(self,batch,batch_idx):
        x = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        loss = F.mse_loss(x,x_hat,reduction='sum')
        self.log('train_loss',loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=1e-5)
        return optimizer
