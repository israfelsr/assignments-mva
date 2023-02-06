import argparse
import wandb

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything, Trainer

from source.models import SimpleConvModel


def main(args):
    #wandb.init()
    # config
    seed_everything(args.seed, workers=True)
    wandb_logger = WandbLogger(project="dl-practice-tp1", name=args.run_name)

    # Data
    dataset = torchvision.datasets.USPS(
        root=args.data_dir,
        train=True,
        transform=transforms.ToTensor(),
        target_transform=torchvision.transforms.Compose([
            lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10),
            lambda x: x.squeeze(), lambda x: x.type(torch.float)
        ]),
        download=False)

    test_set = torchvision.datasets.USPS(
        root=args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        target_transform=torchvision.transforms.Compose([
            lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10),
            lambda x: x.squeeze(), lambda x: x.type(torch.float)
        ]),
        download=False)

    train_gen, val_gen = random_split(
        dataset, [6000, 1291], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_gen,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    val_loader = DataLoader(val_gen,
                            batch_size=args.batch_size,
                            drop_last=True)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
    )

    # Model
    model = SimpleConvModel(output_dim=args.output_dim,
                            batch_size=args.batch_size,
                            base_model=args.model,
                            num_hidden_layers=args.num_hidden_layers,
                            hidden_dim=args.hidden_dim,
                            learning_rate=args.learning_rate,
                            loss=args.loss,
                            activation=args.activation,
                            optimizer=args.optimizer)

    # Trainer
    trainer = Trainer(accelerator=args.device,
                      max_epochs=args.epochs,
                      logger=wandb_logger,
                      default_root_dir="../logs/")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Practice TP1.1")
    # Data args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/israfelsalazar/Documents/mva/dl-in-practice/tp-1/USPS/",
        help="Path to the data folder.")
    # config
    parser.add_argument("--run_name", type=str, help="Run name for wandb")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch Size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Num workers in dataloader")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    # Model
    parser.add_argument("--output_dim",
                        type=int,
                        default=10,
                        help="Number of classes in output")
    parser.add_argument("--hidden_dim",
                        type=int,
                        help="Neurons in hidden layer")
    parser.add_argument("--num_hidden_layers",
                        type=int,
                        help="Numbers of hidden layers")
    parser.add_argument("--activation",
                        default='relu',
                        type=str,
                        help="Activation Function")
    parser.add_argument("--loss",
                        default='mse',
                        type=str,
                        help="Loss Function")
    parser.add_argument("--model",
                        default='linear',
                        type=str,
                        required=True,
                        help="Base architecture model")
    # Trainer
    parser.add_argument("--optimizer",
                        default='sgd',
                        type=str,
                        help="Optimizer to use")
    parser.add_argument("--epochs",
                        type=int,
                        required=True,
                        help="Number of Epochs to train")
    parser.add_argument("--learning_rate",
                        default=0.1,
                        type=float,
                        help="Learning rate")

    args = parser.parse_args()
    main(args)