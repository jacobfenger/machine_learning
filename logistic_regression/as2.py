#########################################################################
# CS 434 Assignment 2 - Logistic Regression
# Jacob Fenger
# Spike Madden
# 4/17/2017
#########################################################################
import csv
import numpy as np
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

def main():

    img_train, truth_train = read_data('usps-4-9-train.csv')

    # An example way to show an image
    #plt.imshow(img_train[0], interpolation='None')
    #plt.show()

if __name__ == "__main__":
    main()
