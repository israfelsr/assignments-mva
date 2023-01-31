import scipy.sparse as sp
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, BertTokenizerFast

from graph_challenge.dataset.utils import *


class GraphProteinDataset(Dataset):

    def __init__(self, adj, features, edges, labels=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.edges = edges

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        adj = self.adj[idx]
        features = self.features[idx]
        edges = self.edges[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return adj, features, edges, y
        else:
            return adj, features, edges


class ProteinDataset(Dataset):

    def __init__(self, data, seq_tokenizer):
        adj, features, edges, sequences, labels = data
        self.adj = adj
        self.features = features
        self.edges = edges
        self.labels = labels

        self.encodings = seq_tokenizer(
            sequences,
            is_split_into_words=True,
            return_offsets_mapping=False,
            truncation=True,
            padding=True,
        )

    def __getitem__(self, idx):
        seq = {key: val[idx] for key, val in self.encodings.items()}
        adj = self.adj[idx]
        features = self.features[idx]
        edges = self.edges[idx]
        labels = self.labels[idx]

        return adj, features, edges, seq, labels

    def __len__(self):
        return len(self.labels)


def collate_graph_batch(batch):
    adj_batch = list()
    features_batch = list()
    #edges_batch = list()
    adj_dist_batch = list()
    idx_batch = list()
    y_batch = list()
    for t, (adj, features, adj_dist, y) in enumerate(batch):
        n = adj.shape[0]
        adj_batch.append(adj + sp.identity(n))
        features_batch.append(features)
        #edges_batch.append(np.sum(edges, axis=0))
        adj_dist_batch.append(adj_dist + sp.identity(n))
        idx_batch.extend([t] * n)
        y_batch.append(y)
    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)
    adj_dist_batch = sp.block_diag(adj_dist_batch)
    return {
        'x': torch.FloatTensor(features_batch),
        'adj': sparse_mx_to_torch_sparse_tensor(adj_batch),
        'adj_dist': sparse_mx_to_torch_sparse_tensor(adj_dist_batch),
        'index': torch.LongTensor(idx_batch),
        'y': torch.LongTensor(y_batch),
    }


def collate_protein_batch(batch):
    adj_batch = list()
    features_batch = list()
    idx_batch = list()
    y_batch = list()

    # seq
    input_ids_batch = list()
    attention_mask_batch = list()
    token_type_ids = list()

    for t, (adj, features, _, seq, y) in enumerate(batch):

        input_ids_batch.append(seq["input_ids"])
        attention_mask_batch.append(seq["attention_mask"])
        token_type_ids.append(seq["token_type_ids"])

        n = adj.shape[0]
        adj_batch.append(adj + sp.identity(n))
        features_batch.append(features)
        idx_batch.extend([t] * n)
        y_batch.append(y)
    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)
    batch = {
        'x': torch.FloatTensor(features_batch),
        'adj': sparse_mx_to_torch_sparse_tensor(adj_batch),
        'index': torch.LongTensor(idx_batch),
        'y': torch.LongTensor(y_batch),
        'seq': {
            'input_ids': torch.tensor(input_ids_batch),
            'attention_mask': torch.tensor(attention_mask_batch),
            'token_type_ids': torch.tensor(token_type_ids)
        }
    }
    return batch


def collate_protein_test_batch(batch):
    adj_batch = list()
    features_batch = list()
    idx_batch = list()

    # seq
    input_ids_batch = list()
    attention_mask_batch = list()
    token_type_ids = list()

    for t, (adj, features, _, seq, _) in enumerate(batch):

        input_ids_batch.append(seq["input_ids"])
        attention_mask_batch.append(seq["attention_mask"])
        token_type_ids.append(seq["token_type_ids"])

        n = adj.shape[0]
        adj_batch.append(adj + sp.identity(n))
        features_batch.append(features)
        idx_batch.extend([t] * n)

    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)
    batch = {
        'x': torch.FloatTensor(features_batch),
        'adj': sparse_mx_to_torch_sparse_tensor(adj_batch),
        'index': torch.LongTensor(idx_batch),
        'seq': {
            'input_ids': torch.tensor(input_ids_batch),
            'attention_mask': torch.tensor(attention_mask_batch),
            'token_type_ids': torch.tensor(token_type_ids)
        }
    }
    return batch


class GraphProteinDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        collate_fn: object = collate_protein_batch,
        batch_size: int = 64,
        train_split: int = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_dir = data_dir
        self.collate_fn = collate_fn

        self.seq_transform = BertTokenizerFast.from_pretrained(
            model_name, do_lower_case=False)

    def setup(self, stage=None):
        train_set, test_set, proteins_test = load_data(self.data_dir)
        train_set, val_set = split_dataset(train_set, self.train_split)
        self.train_gen = ProteinDataset(train_set,
                                        seq_tokenizer=self.seq_transform)
        self.val_gen = ProteinDataset(val_set,
                                      seq_tokenizer=self.seq_transform)
        self.test_gen = ProteinDataset(test_set,
                                       seq_tokenizer=self.seq_transform)
        self.proteins_test = proteins_test

    def train_dataloader(self, ):
        return DataLoader(self.train_gen,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self, ):
        return DataLoader(self.val_gen,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn)

    def test_dataloader(self, ):
        return DataLoader(self.test_gen,
                          batch_size=self.batch_size,
                          collate_fn=collate_protein_test_batch)
