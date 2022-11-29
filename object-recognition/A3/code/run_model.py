#import os
#from tqdm import tqdm
#from omegaconf import OmegaConf

import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.optim as optim
from torchvision import datasets
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything, Trainer
#from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import data_transforms
from definitions import BirdClassifierArguments
from model import BirdClassifierLightningModule, NetLightningModule
from utils import build_config


def main():
    wandb_logger = WandbLogger(name="Assignment3")
    config: BirdClassifierArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    # Setting up the datamodule
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.datasets.train_dir,
                             transform=data_transforms),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.datasets.val_dir,
                             transform=data_transforms),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers)

    print(len(train_loader))
    print(f"samples = {len(train_loader) * config.training.batch_size}")

    # Creating the model
    model = NetLightningModule(num_classes=config.datasets.num_classes,
                               learning_rate=config.training.learning_rate,
                               momentum=config.training.momentum)

    trainer = Trainer(accelerator="auto",
                      max_epochs=config.training.epochs,
                      logger=wandb_logger,
                      log_every_n_steps=10)
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    main()