# CS 434 Assigment 1 Code
# Jacob Fenger
# Spike Madden

import numpy as np
from numpy.linalg import inv
import sys
import random

def load_data(file):

    file = open(file, 'r')

    X = [] # Stores the features
    Y = [] # Stores the outputs

    # loop through each line in the file
    for line in file:

        # tokenize the line
        line = line.split()

        # convert the line to numbers and not strings
        # NOTE: this line is only valid in Python 2.x
        line = map(float, line)

        # Fill the X matrix and append 1 as the dummy variable to the first column
        X.append([1] + line[0:13])

        Y.append(line[13])

    # Convert X and Y to numpy matrices
    X = np.matrix(X)
    Y = np.transpose(np.matrix(Y)) # Make Y a column vector

    return X, Y


def compute_optimal_weight_vector(X, Y):

    # Compute transpose of X
    X_t = np.transpose(X)

    w_1 = inv(np.dot(X_t, X))
    w_2 = np.dot(X_t, Y)

    # Compute optimal weight vector
    w = np.dot(w_1, w_2)

    #print("Optimal weight vector: ", w)

    return w

# Compute the a variant of the optimal weight vector
def variant_optimal_weight_vector(X, Y, lam):

    X_t = np.transpose(X)

    xt = np.dot(X_t, X)
    lam_I = np.dot(lam, np.identity(len(xt)))

    w = np.dot(inv(xt + lam_I), np.dot(X_t, Y))

    return w


def compute_sum_of_squared_error(w, X, Y):

    # Compute the SSE value - needs clean up
    SSE = np.dot(np.transpose((Y - np.dot(X, w))), (Y - np.dot(X, w)))

    return float(SSE[0])

# Takes in a random integer a to represent the max range of the artificial feature
def generate_random_feature(x_train, x_test, a):

    distribution = np.random.uniform(0, a, len(x_train))
    x_train = np.c_[x_train, distribution]
    distribution = np.random.uniform(0, a, len(x_test))
    x_test = np.c_[x_test, distribution]

    return x_train, x_test

def main(args):

    # Ensure enough arguments are present
    if len(args) != 3:
        print "Incorrect number of command lind arguments provided."
        print "Usage: <train file name' 'testing file name'"
        return

    # Save file names
    training_file = args[1]
    test_file = args[2]

    # Load the training and testing data
    train_X, train_Y = load_data(training_file)
    test_X, test_Y = load_data(test_file)

    train_SSE = []
    test_SSE = []

    # COMMENTED OUT TO COMPUTE THE VARIANT WEIGHT VECTOR
    # num_features = 10
    # for i in range(num_features):
    #     a = random.randint(0, 100) + 1
    #     train_X, test_X = generate_random_feature(train_X, test_X, a)
    #
    #     # Compute the optimal weight vector from training data
    #     w = compute_optimal_weight_vector(train_X, train_Y)
    #
    #     train_SSE.append(compute_sum_of_squared_error(w, train_X, train_Y))
    #     test_SSE.append(compute_sum_of_squared_error(w, test_X, test_Y))
    #
    # print "Artificial Features #: "
    # print range(num_features)
    # print "Training SSE: "
    # print train_SSE
    # print "Testing SSE: "
    # print test_SSE

    print "Variant Weight Vector: "
    for lamda in [0.01, 0.05, 0.1, 0.5, 1, 5, 100, 10000000]:
        print "Lamda: " + str(lamda)
        w_var = variant_optimal_weight_vector(train_X, train_Y, lamda)
        # print "w_var:"
        # print w_var
        print "Training data: ", compute_sum_of_squared_error(w_var, train_X, train_Y)
        print "Test data: ", compute_sum_of_squared_error(w_var, test_X, test_Y)

if __name__ == "__main__":
    main(sys.argv)
