from itertools import compress
import numpy as np
import os
import scipy.sparse as sp

import torch


def load_data(data_dir, max_length=None):
    graph_indicator = np.loadtxt(os.path.join(data_dir, "graph_indicator.txt"),
                                 dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(os.path.join(data_dir, "edgelist.txt"),
                       dtype=np.int64,
                       delimiter=",")
    edges_inv = np.vstack((edges[:, 1], edges[:, 0]))
    s = edges[:, 0] * graph_indicator.size + edges[:, 1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort, :]
    edges, idx_unique = np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                      shape=(graph_indicator.size, graph_indicator.size))

    x = np.loadtxt(os.path.join(data_dir, "node_attributes.txt"),
                   delimiter=",")
    edge_attr = np.loadtxt(os.path.join(data_dir, "edge_attributes.txt"),
                           delimiter=",")

    A_dist = sp.csr_matrix((edge_attr[:, 0], (edges[:, 0], edges[:, 1])),
                           shape=(graph_indicator.size, graph_indicator.size))

    #edge_attr = np.vstack((edge_attr, edge_attr))
    #edge_attr = edge_attr[idx_sort, :]
    #edge_attr = edge_attr[idx_unique, :]

    adj = []
    features = []
    #edge_features = []
    adj_dist = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n:idx_n + graph_size[i], idx_n:idx_n + graph_size[i]])
        adj_dist.append(A_dist[idx_n:idx_n + graph_size[i],
                               idx_n:idx_n + graph_size[i]])
        #edge_features.append(edge_attr[idx_m:idx_m + adj[i].nnz, :])
        features.append(x[idx_n:idx_n + graph_size[i], :])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    sequences = list()
    with open(os.path.join(data_dir, "sequences.txt"), 'r') as f:
        for line in f:
            sequences.append(line[:-1])
    sequences = [[*seq] for seq in sequences]
    if max_length is not None:
        sequences = [list(seq)[:max_length - 2] for seq in sequences]

    adj = [normalize_adjacency(A) for A in adj]
    adj_dist = [normalize_adjacency(A) for A in adj_dist]

    train_set, test_set, protein_test = split_testset(
        data_dir, (adj, features, adj_dist, sequences))
    return train_set, test_set, protein_test


def split_testset(data_dir, data):
    adj, features, edges, sequences = data
    adj_train = list()
    features_train = list()
    edges_train = list()
    sequences_train = list()
    y_train = list()

    adj_test = list()
    features_test = list()
    edges_test = list()
    sequences_test = list()
    proteins_test = list()
    with open(os.path.join(data_dir, 'graph_labels.txt'), 'r') as f:
        for i, line in enumerate(f):
            t = line.split(',')
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                adj_test.append(adj[i])
                features_test.append(features[i])
                edges_test.append(edges[i])
                sequences_test.append(sequences[i])
            else:
                adj_train.append(adj[i])
                features_train.append(features[i])
                y_train.append(int(t[1][:-1]))
                edges_train.append(edges[i])
                sequences_train.append(sequences[i])
    train_set = (np.array(adj_train), np.array(features_train),
                 np.array(edges_train), sequences_train, np.array(y_train))
    test_set = (adj_test, features_test, edges_test, sequences_test,
                proteins_test)
    return train_set, test_set, proteins_test


def normalize_adjacency(A):
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


def split_dataset(data, percentage=0.8):
    adj, features, edges, sequences, labels, = data
    mask = np.random.rand(len(labels)) < percentage

    adj_train = adj[mask]
    features_train = features[mask]
    edges_train = edges[mask]
    sequences_train = list(compress(sequences, mask))
    labels_train = labels[mask]
    train_set = (adj_train, features_train, edges_train, sequences_train,
                 labels_train)

    adj_test = adj[~mask]
    features_test = features[~mask]
    edges_test = edges[~mask]
    sequences_test = list(compress(sequences, ~mask))
    labels_test = labels[~mask]
    test_set = (adj_test, features_test, edges_test, sequences_test,
                labels_test)

    return train_set, test_set


def split_dataset(features: np.array,
                  labels: np.array,
                  percentage: float = 0.8):
    
    mask = np.random.rand(len(labels)) < percentage

    X_train = features[mask]
    y_train = labels[mask]
    X_test = features[~mask]
    y_test = labels[~mask]

    return X_train, y_train, X_test, y_test