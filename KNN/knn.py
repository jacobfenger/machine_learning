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

    #features = normalize_features(features, 0, 1)

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

def run_K_nearest_neighbor(train_truth, train_ftrs, test_truth, test_ftrs):

    k = [i for i in range(1, 53, 2)]
    training_mistakes = []
    testing_mistakes = []
    fold_mistakes = []

    for i in k:

        fold_mistakes.append(leave_one_out_validation(train_ftrs, train_truth, i))

        training_mistakes.append(compute_knn_accuracy(i, train_truth, train_ftrs, train_ftrs, train_truth))

        testing_mistakes.append(compute_knn_accuracy(i, train_truth, train_ftrs, test_ftrs, test_truth))

    graph_data(k, fold_mistakes, training_mistakes, testing_mistakes)

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

def _compute_info_gain(greater, smaller):

    greater_correct = 0
    smaller_correct = 0
    total_data_points = len(greater) + len(smaller)

    for x in greater:
        if x[0] == 1:
            greater_correct += 1

    for y in smaller:
        if y[0] == -1:
            smaller_correct += 1

    greater_wrong = len(greater) - greater_correct
    smaller_wrong = len(smaller) - smaller_correct

    information_gain = 0
    greater_information_gain = 0
    smaller_information_gain = 0

    if greater_correct == 0 or greater_wrong == 0:
        greater_information_gain = 0
    else:
        greater_information_gain = -1 * float(greater_correct)/len(greater) * math.log(float(greater_correct)/len(greater), 2) - float(greater_wrong)/len(greater) * math.log(float(greater_wrong)/len(greater), 2)

    if smaller_correct == 0 or smaller_wrong == 0:
        smaller_information_gain = 0
    else:
        smaller_information_gain = -1 * float(smaller_correct)/len(smaller) * math.log(float(smaller_correct)/len(smaller), 2) - float(smaller_wrong)/len(smaller) * math.log(float(smaller_wrong)/len(smaller), 2)

    return float(len(greater))/total_data_points * greater_information_gain, float(len(smaller))/total_data_points * smaller_information_gain

# Returns information to be stored in a node of the decision tree
def get_best_feature(data, parent_gain):

    best_gain = 0
    best_boundary = 0
    best_feature_index = -1
    left = []
    right = []

    for feature in range(1,len(data[0])):

        boundaries = [float(data[i][feature]) for i in range( len(data))]

        for i in range(len(boundaries)):

            greater = []
            smaller = []

            for j in range(len(boundaries)):
                if boundaries[j] >= boundaries[i]:
                    greater.append(data[j])
                elif boundaries[j] < boundaries[i]:
                    smaller.append(data[j])

            g_gain, l_gain = _compute_info_gain(greater, smaller)
            gain = parent_gain - g_gain - l_gain

            if gain > best_gain:
                best_gain = gain
                best_boundary = boundaries[i]
                best_feature_index = feature
                right = greater
                left = smaller

    return {'index': best_feature_index, 'value': best_boundary,
            'left': left, 'right': right, 'gain': (best_gain, g_gain, l_gain)}

# Takes in the current node, the maximum depth, and the current depth
def build_tree(node, max_depth, depth):

    if depth >= max_depth:
        node['left'] = majority_class(node['left'])
        node['right'] = majority_class(node['right'])
        return

    if not node['left'] or not node['right']:
        node['left'] = node['right'] = majority_class(node['left'] + node['right'])
        return

    node['left'] = get_best_feature(node['left'], node['gain'][2])
    build_tree(node['left'], max_depth, depth+1)

    node['right'] = get_best_feature(node['right'], node['gain'][1])
    build_tree(node['right'], max_depth, depth+1)

# Compute the majority class based on the input data group
# Returns 1 or -1
def majority_class(data_group):
    num = 0

    for item in data_group:
        num = num + item[0]

    if num >= 0:
        return 1
    else:
        return -1

def init_tree(max_depth, data):

    root = get_best_feature(data, 1)
    build_tree(root, max_depth, 1)

    return root

def print_tree(node, depth):

    if isinstance(node, dict):
        print('%s[F%d < %.3f] => Info Gain: %.3f' % ((depth*' ', (node['index']), node['value'], node['gain'][0])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print (depth)*' ' + "Class: "+ str(node)

# Special thanks to Jason Brownlee @ Machinelearningmastery.com
# for some advice on an efficient prediction method :)
def predict(node, data_row):

    if data_row[node['index']-1] >= node['value']:
        if isinstance(node['right'], dict):
            return predict(node['right'], data_row)
        else:
            return node['right']
    else:
        if isinstance(node['left'], dict):
            return predict(node['left'], data_row)
        else:
            return node['left']

def classify(tree, data_set, truth_set):
    classification = []
    correct = 0

    for row in data_set:
        classification.append(predict(tree, row))

    for i in range(len(classification)):
        if classification[i] == truth_set[i]:
            correct += 1

    return 1 - float(correct)/len(truth_set)

def main():

    train_truth, train_ftrs = read_data('knn_train.csv')
    test_truth, test_ftrs = read_data('knn_test.csv')

    #run_K_nearest_neighbor(train_truth, train_ftrs, test_truth, test_ftrs)

    # Regroup data set
    train_data = np.insert(train_ftrs, 0, train_truth, axis=1)

    stump = init_tree(1, train_data)
    #print_tree(stump, 0)

    error = classify(stump, train_ftrs, train_truth)
    print "TRAINING ERROR FOR STUMP: ", error
    error = classify(stump, test_ftrs, test_truth)
    print "TESTING ERROR FOR STUMP: ", error

    depth_6_tree = init_tree(6, train_data)
    print_tree(depth_6_tree, 0)

    error = classify(depth_6_tree, train_ftrs, train_truth)
    print "TRAINING ERROR FOR DEPTH 6 TREE: ", error
    error = classify(depth_6_tree, test_ftrs, test_truth)
    print "TESTING ERROR FOR DEPTH 6 TREE: ", error



if __name__ == '__main__':
    main()
