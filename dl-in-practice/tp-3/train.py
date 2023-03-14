import argparse
import wandb

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader


class GATModel(nn.Module):

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.gats = nn.ModuleList()
        self.gats.append(
            graphnn.GATConv(input_dim,
                            hidden_dim,
                            heads=num_heads,
                            dropout=dropout))
        for i in range(num_layers):
            self.gats.append(
                graphnn.GATConv(hidden_dim * num_heads,
                                hidden_dim,
                                heads=num_heads,
                                dropout=dropout))
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "leaky":
            self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, x, adj):
        for i in range(len(self.gats)):
            x = self.gats[i](x, adj)
            x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def train(model, loss_fcn, device, optimizer, max_epochs, train_dataloader,
          val_dataloader):

    epoch_list = []
    scores_list = []

    # loop over epochs
    for epoch in range(max_epochs):
        model.train()
        losses = []
        # loop over batches
        for i, train_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_batch_device = train_batch.to(device)
            # logits is the output of the model
            logits = model(train_batch_device.x, train_batch_device.edge_index)
            # compute the loss
            loss = loss_fcn(logits, train_batch_device.y)
            # optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        wandb.log({"epoch": epoch + 1, "train/loss": loss_data})

        if epoch % 5 == 0:
            # evaluate the model on the validation set
            # computes the f1-score (see next function)
            score = evaluate(model, loss_fcn, device, val_dataloader)
            #print("F1-Score: {:.4f}".format(score))
            wandb.log({"train/f1": score})
            scores_list.append(score)
            epoch_list.append(epoch)

    return epoch_list, scores_list


def evaluate(model, loss_fcn, device, dataloader):

    score_list_batch = []

    model.eval()
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        loss_test = loss_fcn(output, batch.y)
        predict = np.where(output.detach().cpu().numpy() >= 0, 1, 0)
        score = f1_score(batch.y.cpu().numpy(), predict, average="micro")
        score_list_batch.append(score)

    return np.array(score_list_batch).mean()


def main(args):
    wandb.init(project="uncategorized")

    # Train Dataset
    train_dataset = PPI(root="", split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    # Val Dataset
    val_dataset = PPI(root="", split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    # Test Dataset
    test_dataset = PPI(root="", split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Number of features and classes
    n_features, n_classes = train_dataset[0].x.shape[1], train_dataset[
        0].y.shape[1]

    print("Number of samples in the train dataset: ", len(train_dataset))
    print("Number of samples in the val dataset: ", len(test_dataset))
    print("Number of samples in the test dataset: ", len(test_dataset))
    print("Output of one sample from the train dataset: ", train_dataset[0])
    print("Edge_index :")
    print(train_dataset[0].edge_index)
    print("Number of features per node: ", n_features)
    print("Number of classes per node: ", n_classes)

    student_model = GATModel(num_heads=4,
                             num_layers=args.num_hidden_layers,
                             input_dim=n_features,
                             hidden_dim=args.hidden_dim,
                             output_dim=n_classes,
                             dropout=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    student_model.to(device=device)

    ### DEFINE LOSS FUNCTION AND OPTIMIZER
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(),
                                 lr=args.learning_rate)

    ### TRAIN
    epoch_list, student_model_scores = train(student_model, loss_fcn, device,
                                             optimizer, args.epochs,
                                             train_dataloader, val_dataloader)

    score_test = evaluate(student_model, loss_fcn, device, test_dataloader)
    wandb.log({"test/f1": score_test})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Practice TP3")
    # Data args
    parser.add_argument("--run_name", type=str, help="Run name for wandb")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="Batch Size.")
    parser.add_argument("--num_hidden_layers",
                        type=int,
                        default=3,
                        help="Numbers of hidden layers")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=350,
                        help="Neurons in hidden layer")
    parser.add_argument("--activation",
                        default='relu',
                        type=str,
                        help="Activation Function")
    # Trainer
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of Epochs to train")
    parser.add_argument("--learning_rate",
                        default=0.005,
                        type=float,
                        help="Learning rate")

    args = parser.parse_args()
    main(args)