import numpy as np
from numpy.linalg import inv

def load_training_data():

    file = open('housing_train.txt', 'r')
    
    X = [] # Stores the features (including a dummy variable the first column)
    Y = [] # Stores the outputs

    # loop through each line in the file
    for line in file:
        
        # tokenize the line
        line = line.split() 

        # convert the line to numbers and not strings
        # NOTE: this line is only valid in Python 2.x
        line = map(float, line)

        X.append([1, line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11], line[12]])
        
        Y.append(line[13])
    
    # Convert X and Y to numpy matrices
    X = np.matrix(X)
    Y = np.transpose(np.matrix(Y)) # Make Y a column vector

    print Y

    return X, Y

def compute_optimal_weight_vector(X, Y):

    X_t = np.transpose(X)
    
    w_1 = inv(np.dot(X_t, X))
    w_2 = np.dot(X_t, Y)
    
    w = np.dot(w_1, w_2)

    print("Optimal weight vector: ", w)

def main():
    
    # Load the training data 
    X, Y = load_training_data();
    
    # Compute the optimal weight vector
    compute_optimal_weight_vector(X, Y)



if __name__ == "__main__":
    main()
