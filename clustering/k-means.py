######################################################
#                CS 434 Assignment 4
#                   Spike Madden
#                   Jacob Fenger
######################################################
import numpy as np
import math
from matplotlib import pyplot as plt
from itertools import izip
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram

def get_data(filename):
    file = open(filename, 'r')

    data = []

    for line in file:
        data.append(map(float, line.rstrip('\n').split(',')))

    return np.asarray(data)

# Compute euclidean distance between two points, a and b
def _euclidean_distance(a, b):
    return distance.euclidean(a, b)

def k_means(K, data, iters):

    centers = list()
    mins = [0, 0]
    sse = list()
    old_centers = list()

    for i in range(K):
        centers.append(data[np.random.choice(len(data), replace=False)])

    for iters in range(iters):
        clusters = [ [] for _ in range(K) ]

        for x_i in data:
            mins[0] = _euclidean_distance(x_i, centers[0])
            mins[1] = 0 # Set default cluster number to 0

            for c in range(K - 1):
                d = _euclidean_distance(x_i, centers[c + 1])

                if d < mins[0]:
                    mins[0] = d # store current minimum for data point x_i
                    mins[1] = c + 1 # store cluster number

            clusters[mins[1]].append(x_i)

        for i in clusters:
            if len(i) == 0:
                i.append(data[np.random.choice(1, replace=False)])

        for i in range(K):
            centers[i] = np.mean(clusters[i], axis=0)

        # Part 1
        #sse.append(compute_SSE(K, data, clusters, centers))

        sse = compute_SSE(K, data, clusters, centers)


    # plt.plot(range(iters + 1), sse)
    # plt.title('Graph of SSE versus number of iterations for K=2')
    # plt.xlabel('Iteration Number')
    # plt.ylabel('SSE')
    # plt.show()
    return sse


"""
Computes SSE for each cluster
"""
def compute_SSE(K, data, clusters, centers):
    sse = 0
    for k in range(K):
        x_sum = 0
        for x in clusters[k]:
            x_sum += _euclidean_distance(x, centers[k]) ** 2

        sse += x_sum

    #print "SSE: " + str(sse)
    return sse

def main():
    train_data = get_data('data-1.txt')
    sse = list()

    # Part 1
    #k_means(2, train_data, 20);

    # Part 2
    # for k in range(2, 15):
    #     sse.append(k_means(k, train_data, 10))
    #
    # plt.plot(range(2, 15), sse)
    # plt.title('Graph of SSE versus number of clusters')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('SSE')
    # plt.show()

    # Part 3/4
    plt.title('Single Link Dendrogram for Last 10 Clusters')
    plt.xlabel('Cluster number')
    plt.ylabel('Distance')

    Z = linkage(train_data, method='single')

    d = dendrogram(Z, p=10, truncate_mode = 'lastp')
    plt.plot()
    plt.show()


if __name__ == '__main__':
    main()
