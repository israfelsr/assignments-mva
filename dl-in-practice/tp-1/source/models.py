import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl

from source.helpers import Losses, Activations, Optimizers


class SimpleConvModel(pl.LightningModule):

    def __init__(
        self,
        output_dim: int,
        batch_size: int,
        base_model: str,
        num_hidden_layers: int,
        input_dim: int = 256,
        hidden_dim: int = 100,
        learning_rate: float = 0.1,
        loss: str = 'mse',
        activation: str = 'relu',
        optimizer: str = 'sgd',
    ):
        super().__init__()
        self.base_model = Models[base_model](input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             num_layers=num_hidden_layers,
                                             activation=activation)
        if base_model == 'linear':
            fc_dim = hidden_dim
        if base_model == 'conv':
            fc_dim = hidden_dim * input_dim
        self.fc = nn.Linear(fc_dim, output_dim)
        self.learning_rate = learning_rate
        self.loss = Losses[loss]
        self.batch_size = batch_size
        self.metrics = Accuracy(task="multiclass", num_classes=output_dim)
        self.loss_type = loss
        self.optimizer = optimizer

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        # We need to modify this thing with different datasets
        return Optimizers[self.optimizer](self.parameters(),
                                          lr=self.learning_rate)
        #return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        scores = self.forward(x)
        loss = self.loss(scores, y)
        _, predicted = torch.max(scores.data, 1)
        accuracy = self.metrics(predicted, torch.argmax(y, dim=1))

        self.log("loss/train",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        self.log("accuracy/train",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        scores = self.forward(x)
        loss = self.loss(scores, y)
        _, predicted = torch.max(scores.data, 1)
        accuracy = self.metrics(predicted, torch.argmax(y, dim=1))

        self.log("loss/val",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        self.log("accuracy/val",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        scores = self.forward(x)
        loss = self.loss(scores, y)
        _, predicted = torch.max(scores.data, 1)
        accuracy = self.metrics(predicted, torch.argmax(y, dim=1))

        self.log("loss/test",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=self.batch_size)
        self.log("accuracy/test",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=self.batch_size)
        return loss


class LinearModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 activation: str):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_block = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.activation = Activations[activation]

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        for i in range(len(self.fc_block)):
            x = self.activation(self.fc_block[i](x))
        return x


class ConvolutionalModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 activation: str):
        super().__init__()
        self.conv1 = nn.Conv2d(1,
                               hidden_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.conv_block = nn.ModuleList([
            nn.Conv2d(hidden_dim,
                      hidden_dim,
                      kernel_size=5,
                      stride=1,
                      padding=2) for _ in range(num_layers)
        ])
        self.activation = Activations[activation]

    def forward(self, x):
        x = self.activation(self.conv1(x))
        for i in range(len(self.conv_block)):
            x = self.activation(self.conv_block[i](x))
        x = x.view(x.shape[0], -1)
        return x


Models = {'conv': ConvolutionalModel, 'linear': LinearModel}
