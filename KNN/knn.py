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
import itertools

# Normalize the features so they have the same range of values ([0, 1])
# Will return the list of normalize features
def normalize_features(features, min_range, max_range):

    return features / features.max(axis=0)

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

        if truth_value == classification:
            count += 1

    return float(count)/len(feature_set)

def compute_knn_accuracy(k, feature_truth, feature_set, test_set, test_truth):

    count = 0

    for j in range(len(test_set)):
        cl = k_nearest_neighbor(k, feature_truth, feature_set, test_set[j])

        if cl != test_truth[j]:
            count += 1

    return (1 - float(count)/len(test_truth), count)

def main():

    train_truth, train_ftrs = read_data('knn_train.csv')
    test_truth, test_ftrs = read_data('knn_test.csv')

    k = [i for i in range(1, 31, 2)]
    training_error = []
    testing_error = []
    fold_error = []
    incorrect_count = []

    for i in k:

        fold_error.append(leave_one_out_validation(train_ftrs, train_truth, i))

        training_error.append(compute_knn_accuracy(i, train_truth, train_ftrs, train_ftrs, train_truth)[0])

        accuracy, errors = compute_knn_accuracy(i, train_truth, train_ftrs, test_ftrs, test_truth)

        testing_error.append(accuracy)
        incorrect_count.append(errors)


    print "TRAINING: "
    print training_error

    print "TESTING: "
    print testing_error
    print incorrect_count

    print "FOLDING: "
    print fold_error

if __name__ == '__main__':
    main()
