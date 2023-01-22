import wandb

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

from graph_challenge.model import GraphSequenceModel
from graph_challenge.dataset.utils import load_data, split_dataset
from graph_challenge.dataset.datasets import GraphProteinDataModule


def main():
    # Load Data
    epochs = 50
    batch_size = 64
    n_hidden = 64
    n_input = 86
    dropout = 0.2
    learning_rate = 0.001
    n_class = 18
    model_name = 'Rostlab/prot_bert_bfd'
    root = "/Users/israfelsalazar/Documents/mva/ALTEGRAD/graph-challenge/data.nosync"

    wandb_logger = WandbLogger(name="altegrad")
    #if config.training.seed != -1:
    seed_everything(42, workers=True)

    # Create Datamodules
    datamodule = GraphProteinDataModule(
        data_dir=root,
        model_name=model_name,
    )
    datamodule.setup()

    loader = datamodule.train_dataloader()

    # Create Model
    model = GraphSequenceModel(input_dim=n_input,
                               output_dim=n_class,
                               dropout=dropout,
                               model_name=model_name,
                               learning_rate=learning_rate)

    # Train
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        #max_epochs=1,
        logger=wandb_logger,
        #callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()