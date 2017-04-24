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

    classification = []


    vote = 0

    neighbors = find_neighbors(K, training_set, test_point)

    for x in neighbors:
        vote += truth[x[1]]

    # Determine how to classify
    if vote >= 0:
        return 1
    else:
        return -1

def compute_error(classification, truth_set):
    count = 0

    for i in range(len(classification)):
        if classification[i] != truth_set[i]:
            count += 1

    return count

def compute_accuracy(classification, truth_set):
    count = 0

    for i in range(len(classification)):
        if classification[i] == truth_set[i]:
            count += 1

    return float(count)/float(len(truth_set))

def main():

    train_truth, train_ftrs = read_data('knn_train.csv')
    test_truth, test_ftrs = read_data('knn_test.csv')

    k = [i for i in range(1, 53, 2)]
    training_error = []
    testing_error = []
    fold_error = []


    for i in k:
        count = 0

        # Iterate through each of the folds
        for s in range(len(train_ftrs)):

            train_ftrs_split = copy.deepcopy(train_ftrs)
            train_truth_split = copy.deepcopy(train_truth)
            validation_list = []
            truth_value = 0

            validation_list = train_ftrs_split[s]
            truth_value = train_truth_split[s]

            train_ftrs_split = np.delete(train_ftrs_split, s, 0)
            train_truth_split = np.delete(train_truth_split, s, 0)

            classification = k_nearest_neighbor(i, train_truth_split, train_ftrs_split, validation_list)

            if truth_value == classification:
                count += 1

        fold_error.append(float(count)/len(train_ftrs))

        count = 0

        for j in range(len(train_ftrs)):
            cl = k_nearest_neighbor(i, train_truth, train_ftrs, train_ftrs[j])

            if cl == train_truth[j]:
                count += 1

        training_error.append(float(count)/len(train_ftrs))

        count = 0

        for j in range(len(test_ftrs)):
            cl = k_nearest_neighbor(i, train_truth, train_ftrs, test_ftrs[j])

            if cl == test_truth[j]:
                count += 1

        testing_error.append(float(count)/len(train_ftrs))

    print "TRAING: "
    print training_error

    print "TESTING: "
    print testing_error

    print "FOLDING: "
    print fold_error

if __name__ == '__main__':
    main()
