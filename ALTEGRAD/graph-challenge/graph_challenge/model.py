import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl

from transformers import AutoModelForTokenClassification


class GraphNeuralNetwork(pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        adam_betas: float = 0.9,
        adam_eps: float = 0.9,
        adam_weight_decay: float = 0.9,
    ):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.metrics = Accuracy(task="multiclass", num_classes=output_dim)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adam_betas = [adam_betas, adam_betas]
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay

    def forward(self, x_in, adj, idx):
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))

        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)

        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        return F.log_softmax(out, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        scores = self.forward(batch['x'], batch['adj'], batch['index'])
        loss = self.loss(scores, batch['y'])
        accuracy = self.metrics(scores, batch['y'])

        self.log("train/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=64)
        self.log("train/accuracy/classification",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=64)
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self.forward(batch['x'], batch['adj'], batch['index'])
        loss = self.loss(scores, batch['y'])
        accuracy = self.metrics(scores, batch['y'])

        self.log("val/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=64)
        self.log("val/accuracy/classification",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=64)
        return loss


class GraphSequenceModel(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dropout: int,
                 model_name: str,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 adam_betas: float = 0.9,
                 adam_eps: float = 0.9,
                 adam_weight_decay: float = 0.9,
                 freeze_encoder: bool = True):
        super(GraphSequenceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + 1024, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=output_dim)
        self.model = model.bert
        if freeze_encoder:
            for params in self.model.parameters():
                params.requires_grad = False

        self.metrics = Accuracy(task="multiclass", num_classes=output_dim)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adam_betas = [adam_betas, adam_betas]
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay

    def forward(self, batch):
        x_in = batch["x"]
        adj = batch["adj"]
        idx = batch["idx"]
        seq = batch["seq"]

        # sequence embedings
        seq_embeddings = torch.sum(self.model(**seq))

        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))

        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)

        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = torch.cat((out, seq_embeddings))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)

        return F.log_softmax(out, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        loss = self.loss(scores, batch['y'])
        accuracy = self.metrics(scores, batch['y'])

        self.log("train/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=64)
        self.log("train/accuracy/classification",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=64)
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self.forward(batch)
        loss = self.loss(scores, batch['y'])
        accuracy = self.metrics(scores, batch['y'])

        self.log("val/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True,
                 batch_size=64)
        self.log("val/accuracy/classification",
                 accuracy,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=64)
        return loss
