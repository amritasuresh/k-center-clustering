# -*- coding: utf-8 -*-
# !/usr/bin/python3
#
#
# Implementation of a 3-approximation algorithm for the offline k-center with outliers
# clustering problem.
#

import random
import operator
import math
import cartopy.crs      as ccrs
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import timeit
import sys


# Reads dataset, stores tweets as [timestamp, latitude, longitude].
def read_tweets(file):
    tweets = []
    with open(file) as input:
        for line in input:
            # tweets.append([float(x) for x in line.split()])
            tweets.append((float(line.split()[0]),
                           float(line.split()[1]),
                           float(line.split()[2])))
    return tweets


# Plots clusters
def plot(clusters, filename):
    map = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0, globe=None))
    map.stock_img()
    for cluster in clusters:
        colourR = (random.random(), random.random(), random.random())
        map.plot([tweet[1] for tweet in cluster],
                 [tweet[2] for tweet in cluster],
                 "o",
                 color=colourR,
                 markersize=1,
                 )
    plt.savefig(filename)
    # plt.show()

def plot_outliers(outliers, filename):
    map = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0, globe=None))
    map.stock_img()
    colourBlack = (0, 0, 0)
    map.plot([coord[1] for coord in outliers],
             [coord[2] for coord in outliers],
             "o",
             color=colourBlack,
             markersize=1,
            )
    plt.savefig(filename)


# Computes distance between two tweets.
def tweet_dist(v1, v2):
    # update dmin, dmax
    return math.sqrt((v2[2] - v1[2]) ** 2
                     + ((v2[1] - v1[1]) ** 2))


# Computes bounds on a tweet set.
# Starts from a random tweet, computes the distance to the rest of the
# tweets. Repeats from farthest/shortest tweet until it gets the same
# two tweets.

def bound(graph, comp):
    rand_nodes = [graph[i] for i in random.sample(range(len(graph)), 3)]

    # Initial bound.
    bnd = tweet_dist(rand_nodes[0], rand_nodes[1])

    # Random starting point.
    v1 = rand_nodes[2]
    v1_prev = [0.0, 0.0, 0.0]
    # Initialise vnext in case we get the bound at first try.
    vnext = [0.0, 0.0, 0.0]
    v2 = [0.0, 0.0, 0.0]

    while v1 != v1_prev:
        for v2 in graph:
            dist = tweet_dist(v1, v2)
            if comp(dist, bnd) and v1 != v2:
                bnd = dist
                vnext = v2
        v1_prev = v1
        v1 = vnext

    return bnd


# Computes r values based on value of epsilon.
def betas(dmin, dmax, epsilon):
    n = (1 + epsilon)
    betas = []
    while n < dmax:
        if n > dmin:
            betas.append(n)
        n = n * (1 + epsilon)
    return betas


# Builds a cluster within 3r and r distance of a given center.
def build_e_j(graph, radius, center):
    cluster = set()
    cluster.add(center)
    for node in graph:
        if tweet_dist(center, node) < 3 * radius:
            cluster.add(node)
    return cluster


# Finds the number of points within r distance of a given center

def build_g_j(graph, radius, center):
    cluster = set()
    cluster.add(center)
    for node in graph:
        if tweet_dist(center, node) < radius:
            cluster.add(node)
    return len(cluster)


# Clustering code.
# Build clusters out of the centers which have the largest cluster.
def clustering(graph, k, betas, outlier_count):
    result = []
    time_when_clustering_started = timeit.default_timer()
    #print("[***] in loop starting now")
    for beta in betas:  # loops through all the possible values of r until it finds the one which can form a cluster with at most z outliers.
        centers = set()
        #print("[***] Total time spent. Beta is updated, and is ", time_when_clustering_started, beta)
        clusters = []
        unclustered = set(graph)
        size_of_graph = len(unclustered)

        nb_centers = 0

        #print(beta, "\t", betas.index(beta) + 1, "out of\t", len(betas),
        #      flush=True)
        time_at_beta_start = timeit.default_timer()

        size_of_clusters = 0

        while nb_centers < k:
            # loop to find max g_j
            time_at_new_loop = timeit.default_timer()-time_when_clustering_started
            #print("[***] Total time spent. We are now at loop, and k is ", time_at_new_loop, k)
            max = 0
            remove_cluster = set()
            unclustered_temp = set(unclustered)
            time_before_building_disks = timeit.default_timer()

            while len(unclustered_temp):
                new_center = unclustered_temp.pop()
                g_j = build_g_j(unclustered, beta, new_center)
                if g_j > max:
                    max_center = new_center
                    max = g_j
                    remove_cluster = build_e_j(unclustered, beta, new_center)

            #print("Loop to find single center took", timeit.default_timer()-time_before_building_disks)
            clusters.append(remove_cluster)
            centers.add(max_center)
            unclustered = set(x for x in unclustered if x not in clusters[-1])
            nb_centers += 1
            size_of_clusters += len(remove_cluster)
            #print("center found", len(remove_cluster))

            #OPTIMIZATION: Stop the loop if it looks like it is going nowhere
            if (k-nb_centers+1)*len(remove_cluster)+outlier_count+size_of_clusters<size_of_graph:
                nb_centers=k+100

        if len(unclustered) < outlier_count + 1:
            result.append([centers, clusters, unclustered, beta])
            break

    return result

# Plots the best solution.
def plot_solution(L, filename):
    plot(L[1], filename)


if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("[*] python3 main.py k epsilon window outliers")
        sys.exit()

    k = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    window = int(sys.argv[3])
    outlier_count = int(sys.argv[4])
    if len(sys.argv) > 6:
        stop = int(sys.argv[6])
    else:
        stop = 0

    time = []
    best_beta = []
    start = timeit.default_timer()
    #print("[*] Parsing input file.")
    checkpoint = timeit.default_timer()

    tweets = read_tweets("dataset/twitter_1000000.txt")

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    #print("[*] Computing dmin.")
    checkpoint = timeit.default_timer()

    dmin = bound(tweets[:window], operator.lt)

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)
    #print("[*] Computing dmax.")
    checkpoint = timeit.default_timer()

    dmax = bound(tweets[:window], operator.gt)

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    #print("[*] Got a dmin of:\t", dmin)
    #print("[*] Got a dmax of:\t", dmax)

    #print("[*] Initial clustering.")
    checkpoint = timeit.default_timer()

    lbetas = betas(dmin, dmax, epsilon)
    L = clustering(tweets[:window], k, lbetas, outlier_count)

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    #print("[*] Found r:\t", L[0][3])
    #print("[*] Found centers: \t", L[0][0])
    #print("[******] Time taken", ctime)
    #print("[******] Radius", L[0][3])
    #print("[******] For k, epsilon, size, outliers", k, epsilon, window, outlier_count)
    filename = "k"+str(k)+"e"+str(epsilon)+"z"+str(outlier_count)+"s"+str(window)+"tweets.jpg"
    plot_solution(L[0], "img/cluster"+filename)
    #plot_outliers(L[0][2], "img/outlier"+filename)
    print(str(k) +"," + str(epsilon) +"," + str(window)+","+str(outlier_count)+","+str(L[0][3])+","+str(ctime))
    #print(time)
