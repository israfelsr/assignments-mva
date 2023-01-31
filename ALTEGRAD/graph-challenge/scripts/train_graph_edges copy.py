import wandb

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

from graph_challenge.model import MultiGNN
from graph_challenge.dataset.utils import load_data, split_dataset
from graph_challenge.dataset.datasets import (GraphProteinDataset,
                                              collate_graph_batch)


def main():
    # Load Data
    epochs = 60
    n_input = 86
    dropout = 0.2
    learning_rate = 0.005
    n_class = 18
    root = "/Users/israfelsalazar/Documents/mva/ALTEGRAD/graph-challenge/data.nosync"

    wandb_logger = WandbLogger(project="altegrad", name="graph-edges")
    #if config.training.seed != -1:
    seed_everything(42, workers=True)

    # Create Datamodules
    train_set, test_set, _ = load_data(root)
    train_set, val_set = split_dataset(train_set)
    adj_train, features_train, edges_train, _, labels_train = train_set
    adj_val, features_val, edges_val, _, labels_val = val_set

    train_gen = GraphProteinDataset(adj_train, features_train, edges_train,
                                    labels_train)
    val_gen = GraphProteinDataset(adj_val, features_val, edges_val, labels_val)

    train_loader = DataLoader(train_gen,
                              batch_size=64,
                              shuffle=True,
                              collate_fn=collate_graph_batch)
    val_loader = DataLoader(val_gen,
                            collate_fn=collate_graph_batch,
                            batch_size=64)

    # Create Model
    model = MultiGNN(input_dim=n_input,
                     dropout=dropout,
                     output_dim=n_class,
                     learning_rate=learning_rate)

    # Train
    trainer = Trainer(accelerator="cpu",
                      max_epochs=epochs,
                      logger=wandb_logger,
                      default_root_dir="../logs/"
                      #callbacks=[checkpoint_callback],
                      )
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == "__main__":
    main()