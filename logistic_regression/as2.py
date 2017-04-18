#########################################################################
# CS 434 Assignment 2 - Logistic Regression
# Jacob Fenger
# Spike Madden
# 4/17/2017
#########################################################################
import csv
import numpy as np
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

            # Reshape the array so it's 16x16
            image = np.reshape(np.asarray(row[0:256]), (16, 16), order='F')

            # Add image and truth file for each row in CSV file
            image_set.append(image)
            truth_set.append(row[256])

    return image_set, truth_set

# Implementation of the batch gradient descent algorithm to train a binar
# logistic regression classifier.
# Inputs: x        -> image dataset
#         y        -> solution dataset
#         lrn_rate -> learning rate to be used
def batch_gradient_descent(x, y, lrn_rate):

    # Used as an ending condition for the algorithm
    iterations = 0
    n = len(x) # Get number of inputs

    y_hat = []
    w = np.reshape([0 for i in range(256)], (16, 16))

    while iterations < 1:

        d = np.reshape([0 for i in range(256)], (16, 16))

        for i in range(n):

            print 1 / (1 + np.exp( (-1 * w) * x[i]))
            break;
            #y_hat.append(1 / (1 + np.exp( np.dot(-1, w) * x[i])))

            error = y[0] - y_hat[0]

            d = d + np.dot(error, x[i])

        w = w + np.dot(lrn_rate, d)

        iterations += 1

def main():

    img_train, truth_train = read_data('usps-4-9-train.csv')

    # An example way to show an image
    #plt.imshow(img_train[2], interpolation='none', cmap='gray')
    #plt.show()

    batch_gradient_descent(img_train, truth_train, .5)

if __name__ == "__main__":
    main()
