import csv
import numpy as np
import os
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

root = "/Users/israfelsalazar/Documents/mva/ALTEGRAD/graph-challenge/data.nosync"
# Read sequences
sequences = list()
with open(os.path.join(root, 'sequences.txt'), 'r') as f:
    for line in f:
        sequences.append(line[:-1])

# Split data into training and test sets
sequences_train = list()
sequences_test = list()
proteins_test = list()
y_train = list()
with open(os.path.join(root, 'graph_labels.txt'), 'r') as f:
    for i, line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            sequences_test.append(sequences[i])
        else:
            sequences_train.append(sequences[i])
            y_train.append(int(t[1][:-1]))

# Map sequences to
vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train = vec.fit_transform(sequences_train)
X_test = vec.transform(sequences_test)

# Train a logistic regression classifier and use the classifier to
# make predictions
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

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
