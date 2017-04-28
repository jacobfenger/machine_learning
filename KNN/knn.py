##################################################
# Assignment 3 Implementation
# Spike Madden
# Jacob Fenger
# 4/22/2017
##################################################
import csv
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
import itertools

# Normalize the features so they have the same range of values ([0, 1])
# Will return the list of normalize features
def normalize_features(features, min_range, max_range):

    return (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

# Read data from a CSV file and then normalize the features
# Returns the truth labels and normalized feature array
def read_data(filename):

    true_labels = []
    features = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            line = map(float, line)
            true_labels.append(line[0])

            # Normalize the features
            features.append(line[1:])

    features = np.asarray(features)
    true_labels = np.asarray(true_labels)

    features = normalize_features(features, 0, 1)

    return true_labels, features

def euclidean_distance(a, b):
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i]) ** 2

    return math.sqrt(d)

# Find the K nearest neighbors to the test values
def find_neighbors(K, training_set, test):
    distances = []

    for i in range(len(training_set)):
        d = euclidean_distance(training_set[i], test)
        distances.append((d, i))

    return sorted(distances)[:K]

# Implementation of the K nearest-neighbor algorithm
# Returns the classification found for the test set
def k_nearest_neighbor(K, truth, training_set, test_point):

    vote = 0

    neighbors = find_neighbors(K, training_set, test_point)

    for x in neighbors:
        vote += truth[x[1]]

    # Determine how to classify
    if vote >= 0:
        return 1
    else:
        return -1

# Implementation of the leave-one-out cross validation
def leave_one_out_validation(feature_set, truth_set, k):

    count = 0

    for i in range(len(feature_set)):

        train_ftrs = copy.deepcopy(feature_set)
        train_truth = copy.deepcopy(truth_set)

        validation_ftr = train_ftrs[i]
        truth_value = truth_set[i]

        train_ftrs = np.delete(train_ftrs, i, 0)
        train_truth = np.delete(truth_set, i, 0)

        classification = k_nearest_neighbor(k, train_truth, train_ftrs, validation_ftr)

        if truth_value != classification:
            count += 1

    return count

def compute_knn_accuracy(k, feature_truth, feature_set, test_set, test_truth):

    count = 0

    for j in range(len(test_set)):
        cl = k_nearest_neighbor(k, feature_truth, feature_set, test_set[j])

        if cl != test_truth[j]:
            count += 1

    return count

def graph_data(k, fold_mistakes, training_mistakes, testing_mistakes):

    trn, = plt.plot(k, training_mistakes, label='Training')
    fld, = plt.plot(k, fold_mistakes, label='Folding')
    tst, = plt.plot(k, testing_mistakes, label='Testing')

    plt.legend([trn, fld, tst], ["Training", "Folding", "Testing"])

    plt.title("Number of Mistakes for Training, Leave-One-Out-Validation, And"
                " Testing")

    plt.ylabel("Number of Mistakes")
    plt.xlabel("Number of Neighbors")

    plt.xlim([1, 51])

    plt.show()

def main():

    train_truth, train_ftrs = read_data('knn_train.csv')
    test_truth, test_ftrs = read_data('knn_test.csv')

    k = [i for i in range(1, 53, 2)]
    training_mistakes = []
    testing_mistakes = []
    fold_mistakes = []

    for i in k:

        fold_mistakes.append(leave_one_out_validation(train_ftrs, train_truth, i))

        training_mistakes.append(compute_knn_accuracy(i, train_truth, train_ftrs, train_ftrs, train_truth))

        testing_mistakes.append(compute_knn_accuracy(i, train_truth, train_ftrs, test_ftrs, test_truth))

    graph_data(k, fold_mistakes, training_mistakes, testing_mistakes)

if __name__ == '__main__':
    main()
