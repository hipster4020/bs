import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

torch.set_printoptions(precision=0)

class PLModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        
        loss = F.mse_loss(y_hat, y)
        loss /= x.size(0)
        
        log_dict = {
            "train/loss": loss,
        }
        self.log_dict(log_dict, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        
        loss = F.mse_loss(y_hat, y)
        loss /= y_hat.size(0)

        log_dict = {
            "eval/loss": loss
        }
        self.log_dict(log_dict, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=self.learning_rate)
        return optimizer
