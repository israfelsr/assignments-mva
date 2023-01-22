import csv

import torch
from graph_challenge.model import GraphSequenceModel
from graph_challenge.dataset import GraphProteinDataModule


def main():
    epochs = 50
    batch_size = 64
    n_hidden = 64
    n_input = 86
    dropout = 0.2
    learning_rate = 0.001
    n_class = 18
    model_name = 'Rostlab/prot_bert_bfd'
    root = "/Users/israfelsalazar/Documents/mva/ALTEGRAD/graph-challenge/data.nosync"
    model = GraphSequenceModel.load_from_checkpoint("../logs/checkpoint.ckpt")
    root = ""
    # disable randomness, dropout, etc...
    model.eval()
    y_pred_proba = list()
    datamodule = GraphProteinDataModule(
        data_dir=root,
        model_name=model_name,
    )
    datamodule.setup()

    for batch in datamodule.test_dataloader:
        output = model(batch)
        y_pred_proba.append(output)

    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_pred_proba = torch.exp(y_pred_proba)
    y_pred_proba = y_pred_proba.detach().cpu().numpy()

    # Write predictions to a file
    with open('sample_submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = list()
        for i in range(18):
            lst.append('class' + str(i))
        lst.insert(0, "name")
        writer.writerow(lst)
        for i, protein in enumerate(datamodule.proteins_test):
            lst = y_pred_proba[i, :].tolist()
            lst.insert(0, protein)
            writer.writerow(lst)
