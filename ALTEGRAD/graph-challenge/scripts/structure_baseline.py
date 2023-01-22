import csv
import time
import numpy as np
import os
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

root = "/Users/israfelsalazar/Documents/mva/ALTEGRAD/graph-challenge/data.nosync"


def load_data():
    """
    Function that loads graphs
    """
    graph_indicator = np.loadtxt(os.path.join(root, "graph_indicator.txt"),
                                 dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(os.path.join(root, "edgelist.txt"),
                       dtype=np.int64,
                       delimiter=",")
    edges_inv = np.vstack((edges[:, 1], edges[:, 0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:, 0] * graph_indicator.size + edges[:, 1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort, :]
    edges, idx_unique = np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(graph_indicator.size, graph_indicator.size))

    x = np.loadtxt(os.path.join(root, "node_attributes.txt"), delimiter=",")
    edge_attr = np.loadtxt(os.path.join(root, "edge_attributes.txt"),
                           delimiter=",")
    edge_attr = np.vstack((edge_attr, edge_attr))
    edge_attr = edge_attr[idx_sort, :]
    edge_attr = edge_attr[idx_unique, :]

    adj = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n + graph_size[i], idx_n:idx_n + graph_size[i]])
        edge_features.append(edge_attr[idx_m:idx_m + adj[i].nnz, :])
        features.append(x[idx_n:idx_n + graph_size[i], :])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features


def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """

    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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


# Load graphs
adj, features, edge_features = load_data()

# Normalize adjacency matrices
adj = [normalize_adjacency(A) for A in adj]

# Split data into training and test sets
adj_train = list()
features_train = list()
y_train = list()
adj_test = list()
features_test = list()
proteins_test = list()
with open(os.path.join(root, 'graph_labels.txt'), 'r') as f:
    for i, line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            adj_test.append(adj[i])
            features_test.append(features[i])
        else:
            adj_train.append(adj[i])
            features_train.append(features[i])
            y_train.append(int(t[1][:-1]))

# Initialize device
device = torch.device("mps") if torch.cuda.is_available() else torch.device(
    "cpu")

# Hyperparameters
epochs = 50
batch_size = 64
n_hidden = 64
n_input = 86
dropout = 0.2
learning_rate = 0.001
n_class = 18

# Compute number of training and test samples
N_train = len(adj_train)
N_test = len(adj_test)

# Initializes model and optimizer
model = GNN(n_input, n_hidden, dropout, n_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Train model
for epoch in range(epochs):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    count = 0
    # Iterate over the batches
    for i in range(0, N_train, batch_size):
        adj_batch = list()
        features_batch = list()
        idx_batch = list()
        y_batch = list()

        # Create tensors
        for j in range(i, min(N_train, i + batch_size)):
            n = adj_train[j].shape[0]
            adj_batch.append(adj_train[j] + sp.identity(n))
            features_batch.append(features_train[j])
            idx_batch.extend([j - i] * n)
            y_batch.append(y_train[j])

        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)
        y_batch = torch.LongTensor(y_batch).to(device)

        optimizer.zero_grad()
        output = model(features_batch, adj_batch, idx_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item()
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print('Epoch: {:03d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(train_loss / count),
              'acc_train: {:.4f}'.format(correct / count),
              'time: {:.4f}s'.format(time.time() - t))

# Evaluate model
model.eval()
y_pred_proba = list()
# Iterate over the batches
for i in range(0, N_test, batch_size):
    adj_batch = list()
    idx_batch = list()
    features_batch = list()
    y_batch = list()

    # Create tensors
    for j in range(i, min(N_test, i + batch_size)):
        n = adj_test[j].shape[0]
        adj_batch.append(adj_test[j] + sp.identity(n))
        features_batch.append(features_test[j])
        idx_batch.extend([j - i] * n)

    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)

    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
    features_batch = torch.FloatTensor(features_batch).to(device)
    idx_batch = torch.LongTensor(idx_batch).to(device)

    output = model(features_batch, adj_batch, idx_batch)
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
    for i, protein in enumerate(proteins_test):
        lst = y_pred_proba[i, :].tolist()
        lst.insert(0, protein)
        writer.writerow(lst)
