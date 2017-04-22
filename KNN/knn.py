##################################################
# Assignment 3 Implementation
# Spike Madden
# Jacob Fenger
# 4/22/2017
##################################################
import csv
import numpy as np

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

def main():

    train_truth, train_ftrs = read_data('knn_train.csv')

if __name__ == '__main__':
    main()
