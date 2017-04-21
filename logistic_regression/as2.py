#########################################################################
# CS 434 Assignment 2 - Logistic Regression
# Jacob Fenger
# Spike Madden
# 4/17/2017
#########################################################################
import csv
import numpy as np
import time
import math
from matplotlib import pyplot as plt

# Read the image data from a given file and store each image in an array
# Returns the images from the CSV as well as the truth values for each image
def read_data(filename):

    image_set = []
    truth_set = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)

        # Loop through each row in CSV file
        for row in reader:
            row = map(float, row)

            # Add image and truth file for each row in CSV file
            image_set.append(row[0:256])
            truth_set.append(row[256])

    return np.asarray(image_set), np.asarray(truth_set)

# Compute the accuracy of the approximated y value to the truth set
def compute_accuracy(y_approx, y_truth):

    correct = 0
    num_images = len(y_truth)

    for i in range(num_images):

        if (round(y_approx[i]) == y_truth[i]):
            correct += 1

    accuracy = float(correct)/float(num_images)

    return accuracy

# Implementation of the batch gradient descent algorithm to train a binary
# logistic regression classifier.
# Inputs: x        -> image dataset
#         y        -> solution dataset
#         lrn_rate -> learning rate to be used
def batch_gradient_descent(x, y, w, lrn_rate, iters, training):

    # Used as an ending condition for the algorithm
    iterations = 0
    n = len(x) # Get number of inputs
    percent_accurate = []

    y_hat = [0 for i in range(n)]

    while iterations < iters:

        d = [0 for i in range(256)]

        for i in range(n):

            y_hat[i] = (1 / (1 + np.exp(np.dot(np.dot(-1, w), x[i]))))

            error = y[i] - y_hat[i]

            d = d + np.dot(error, x[i])

        if training:
            w = w + np.dot(lrn_rate, d)

        acc = compute_accuracy(y_hat, y)
        percent_accurate.append(acc)

        iterations += 1

    return w, [i for i in range(iters)], percent_accurate

# Implementation of the batch gradient descent algorithm to train a binary
# logistic regression classifier.
# Inputs: x        -> image dataset
#         y        -> solution dataset
#         lrn_rate -> learning rate to be used
def modified_batch_gradient_descent(x, y, w, lrn_rate, iters, lamda, training):

    # Used as an ending condition for the algorithm
    iterations = 0
    n = len(x) # Get number of inputs
    percent_accurate = []

    y_hat = [0 for i in range(n)]

    while iterations < iters:

        d = [0 for i in range(256)]

        for i in range(n):

            y_hat[i] = (1 / (1 + np.exp(np.dot(np.dot(-1, w), x[i]))))

            error = y[i] - y_hat[i]

            d = d + np.dot(error, x[i])

        if training:
            w = w + np.dot(lrn_rate, d + np.dot(lamda, w))

        acc = compute_accuracy(y_hat, y)
        percent_accurate.append(acc)

        iterations += 1

    return w, [i for i in range(iters)], percent_accurate


def main():

    img_train, truth_train = read_data('usps-4-9-train.csv')
    img_test, truth_test = read_data('usps-4-9-test.csv')

    learning_rate = 0.0000001

    w = [0 for i in range(256)]

    training = 1

    w, x, y_train, = batch_gradient_descent(img_train, truth_train, w, learning_rate, 100, training)

    training = 0

    w, x, y_test = batch_gradient_descent(img_test, truth_test, w, learning_rate, 100, training)

    plt.plot(x, y_train)
    plt.plot(x, y_test)
    plt.title('Accuracy of batch gradient descent with no regularization\n'
    'given a number of iterations')
    plt.ylabel('Accuracy compared to truth set')
    plt.xlabel('Number of iterations')
    plt.show()

    # PART 4
    # modified batch gradient algorithm
    # with l2 regularization
    #
    # lamdas = [0.01, 0.1, 1, 10, 100, 1000]
    #
    # for lamda in lamdas:
    #     w = [0 for i in range(256)]
    #     training = 1
    #
    #     w, x, y_train, = modified_batch_gradient_descent(img_train, truth_train, w, learning_rate, 100, lamda, training)
    #
    #     training = 0
    #
    #     w, x, y_test = modified_batch_gradient_descent(img_test, truth_test, w, learning_rate, 100, lamda, training)
    #
    #     plt.plot(x, y_train)
    #     plt.plot(x, y_test)
    #     plt.title('Accuracy of batch gradient descent with regularization lamda of %s\n'
    #     'given a number of iterations'%(lamda))
    #     plt.ylabel('Accuracy compared to truth set')
    #     plt.xlabel('Number of iterations')
    #     plt.show()

if __name__ == "__main__":
    main()
