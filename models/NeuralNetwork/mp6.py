# mp6.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import neuralnet as p
import torch

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels,dev_set,dev_labels):
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print("Lengths of predicted labels don't match length of actual labels", len(yhats),len(dev_labels))
        return 0.,0.,0.,0.
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(dev_labels))])
    precision = tp / np.sum([yhats[i]==1 for i in range(len(dev_labels))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(dev_labels))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy,f1,precision,recall

def main(args):
    train_set, train_labels, dev_set,dev_labels = reader.load_dataset(args.dataset_file)
    train_set = torch.tensor(train_set,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.int64)
    dev_set = torch.tensor(dev_set,dtype=torch.float32)
    _,predicted_labels,net = p.fit(train_set,train_labels, dev_set,args.max_iter)
    accuracy,f1,precision,recall = compute_accuracies(predicted_labels,dev_set,dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)
    torch.save(net, "net.model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP6 Neural Net')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = '../data/mp6_data',
                        help='the directory of the training data')
    parser.add_argument('--max_iter',dest="max_iter", type=int, default = 100,
                        help='Maximum iterations - default 10')

    args = parser.parse_args()
    main(args)
