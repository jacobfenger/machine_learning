import numpy as np
from numpy.linalg import inv
import sys

def load_data(file):

    file = open(file, 'r')
    
    X = [] # Stores the features (including a dummy variable the first column)
    Y = [] # Stores the outputs

    # loop through each line in the file
    for line in file:
        
        # tokenize the line
        line = line.split() 

        # convert the line to numbers and not strings
        # NOTE: this line is only valid in Python 2.x
        line = map(float, line)
        
        # Fill the X matrix and append 1 as the dummy variable to the first column
        # Removing the use of the dummy variable results in a higher SSE 
        X.append([1, line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12]])
        
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

def compute_sum_of_squared_error(w, X, Y):
    
    # Compute the SSE value - needs clean up
    SSE = np.dot(np.transpose((Y - np.dot(X, w))), (Y - np.dot(X, w)))
    
    return float(SSE[0])

def main(args):
    
    # Ensure enough arguments are present
    if len(args) != 3:
        print "Incorrect number of command lind arguments provided."
        print "Usage: <train file name' 'testing file name'"
        return
    
    # Save file names
    training_file = args[1]
    test_file = args[2]

    # Load the training data 
    train_X, train_Y = load_data(training_file)
    test_X, test_Y = load_data(test_file)

    # Compute the optimal weight vector from training data
    w = compute_optimal_weight_vector(train_X, train_Y)

    SSE_train = compute_sum_of_squared_error(w, train_X, train_Y)
    SSE_test = compute_sum_of_squared_error(w, test_X, test_Y)    

    print "SSE from the training data: ", SSE_train
    print "SSE from the test data: ", SSE_test

if __name__ == "__main__":
    main(sys.argv)
