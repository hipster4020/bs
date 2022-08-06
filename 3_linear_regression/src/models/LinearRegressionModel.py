import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import functional as FM

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
        
        # y_hat = y_hat.cpu().numpy()
        # y_hat = torch.from_numpy(np.where(y_hat < 0., 0., y_hat)).cuda()
        
        # y_hat = torch.where(y_hat < 0, 0, y_hat)
        # y_hat = y_hat.float()

        # round, loss 확인
        print(f"y_hat : {y_hat[:10]}")
        print(f"y : {y[:10]}")
        
        loss = F.mse_loss(y_hat, y)
        loss /= x.size(0)
        val_acc = FM.accuracy(y_hat.int(), y.int())

        log_dict = {
            "eval/loss": loss,
            "eval/acc": val_acc,
        }
        self.log_dict(log_dict, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=self.learning_rate)
        return optimizer
