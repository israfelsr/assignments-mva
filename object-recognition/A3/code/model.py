import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import torchvision
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

    def __init__(
        self,
        num_classes: int = 20,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        adam_betas: float = 0.9,
        adam_eps: float = 0.9,
        adam_weight_decay: float = 0.9,
    ):
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
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=self.adam_betas,
                                 eps=self.adam_eps,
                                 weight_decay=self.adam_weight_decay)

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

    def __init__(
        self,
        num_classes: int = 20,
        learning_rate: float = 0.001,
        adam_betas: float = 0.9,
        adam_eps: float = 0.9,
        adam_weight_decay: float = 0.9,
    ):
        super().__init__()
        self.base_model = torchvision.models.resnet152()
        checkpoint = torch.load("./pretrained_weights/resnet152-394f9c45.pth")
        self.base_model.load_state_dict(checkpoint)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Linear(in_features=num_ftrs,
                                    out_features=num_classes)
        self.metrics = Accuracy()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=self.adam_betas,
                                 eps=self.adam_eps,
                                 weight_decay=self.adam_weight_decay)

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


class BCNNLightningModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int = 20,
        learning_rate: float = 0.001,
        adam_betas: float = 0.9,
        adam_eps: float = 0.9,
        adam_weight_decay: float = 0.9,
    ):
        super().__init__()
        resnet = torchvision.models.resnet152()
        checkpoint = torch.load("./pretrained_weights/resnet152-394f9c45.pth")
        resnet.load_state_dict(checkpoint)
        self.num_ftrs = resnet.fc.in_features
        layers = list(resnet.children())[:-2]
        self.base_model = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features=self.num_ftrs**2,
                                    out_features=num_classes)
        self.dropout = nn.Dropout(0.5)
        nn.init.xavier_normal_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, val=0)

        self.metrics = Accuracy()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.learning_rate = learning_rate
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.shape[0], self.num_ftrs, -1)
        x = self.dropout(x)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (4)
        x = x.view(x.shape[0], self.num_ftrs**2)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=self.adam_betas,
                                 eps=self.adam_eps,
                                 weight_decay=self.adam_weight_decay)

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