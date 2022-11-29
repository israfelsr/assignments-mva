import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl

nclasses = 20


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# I wanted to practice with pytorch lightning so I will be using it instead of plain pytroch.
class NetLightningModule(pl.LightningModule):

    def __init__(self,
                 num_classes: int,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        super(NetLightningModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.metrics = Accuracy()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=self.momentum)

    def training_step(self, batch, batch_idx):
        image, label = batch
        scores = self(image)
        loss = self.loss(scores, label)
        accuracy = self.metrics(scores, label)
        self.log("train/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True)
        self.log(
            "train/accuracy/classification",
            accuracy,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        scores = self(image)
        loss = self.loss(scores, label)
        accuracy = self.metrics(scores, label)
        self.log("validation/losses/classification",
                 loss,
                 prog_bar=True,
                 logger=True)
        self.log(
            "validation/accuracy/classification",
            accuracy,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss


class BirdClassifierLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass