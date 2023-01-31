import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_with_logits(x, y):
    x = F.softmax(x, dim=1)
    loss = F.mse_loss(x, y)
    return loss


Losses = {'mse': mse_with_logits, 'ce': F.cross_entropy}

Activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

Optimizers = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'rmsprop': torch.optim.RMSprop,
}
