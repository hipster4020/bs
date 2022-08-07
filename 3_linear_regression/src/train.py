import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dataloader import CustomDataLoader, CustomDataset
from models.LinearRegressionModel import PLModel


@hydra.main(config_name="config.yml")
def main(cfg):
    # datasets
    dataset = CustomDataset(cfg.DATASETS.data_path)

    # dataloader
    train_dataloader, eval_dataloader = CustomDataLoader(
        dataset, **cfg.DATALOADER
    ).dataloader()

    # model
    model = PLModel(**cfg.MODEL)

    # logs
    wandb_logger = WandbLogger(**cfg.PATH.wandb)
    callbacks = [ModelCheckpoint(**cfg.PATH.ckpt)]

    # train
    trainer = Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        **cfg.TRAININGARGS,
    )
    trainer.fit(
        model,
        train_dataloader,
        eval_dataloader,
    )

if __name__ == "__main__":
    main()
