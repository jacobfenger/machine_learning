######################################################
#                CS 434 Assignment 4
#                   Spike Madden
#                   Jacob Fenger
######################################################
import math
import sys
import random
import numpy as np

def get_data(filename):
    file = open(filename, 'r')

    data = []

    for line in file:
        line = line.split(',')
        line = map(int, line)
        data.append(line)

    return data

def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2

    return math.sqrt(distance)

def k_means(d, k):
    # select k random samples from D as centers {u1 ... uk}
    centers = random.sample(d, k)

    # k clusters
    clusters = {}

    cluster_changed = 1
    iterations = 0

    # while not converged
    while cluster_changed:
        # reset clusters
        for index in range(k):
            clusters[index + 1] = []

        # assign xi to cj such that d(uj, x1) is minimized
        for point in d:
            min_distance = 100000
            closest_cluster = -1

            for index, centroid in enumerate(centers):
                distance = euclidean_distance(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = index

            # assign point to cluster
            clusters[closest_cluster + 1].append(point)


        cluster_changed = 0

        # update cluster centers
        for key, value in clusters.iteritems():
            a = np.array(value)
            average = np.mean(a, axis = 0)
            if np.all(centers[key - 1]) != np.all(average):
                centers[key - 1] = average
                cluster_changed = 1

        iterations += 1

        print centers
        print len(centers)
        print len(centers[0])

    return clusters, iterations

def main():
    train_data = get_data('data-1.txt')
    result, num = k_means(train_data, 2)

if __name__ == '__main__':
    main()
