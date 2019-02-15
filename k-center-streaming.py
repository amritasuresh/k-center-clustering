import random
import operator
import math
import cartopy.crs      as ccrs
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import timeit
import sys
import itertools


# Reads dataset, stores tweets as [timestamp, latitude, longitude].
def read_tweets (file):
    tweets = []
    with open(file) as input:
        for line in input:
            #tweets.append([float(x) for x in line.split()])
            tweets.append((float(line.split()[0]),
                           float(line.split()[1]),
                           float(line.split()[2])))
    return tweets


# Plots clusters
def plot(clusters, filename):
    map = plt.axes(projection=ccrs.PlateCarree())
    map.stock_img()
    for cluster in clusters:
        colourR = (random.random(), random.random(), random.random())
        map.plot([tweet[1] for tweet in cluster],
                 [tweet[2] for tweet in cluster],
                 "o",
                 color = colourR,
                 markersize = 1,)
    plt.savefig(filename)
    #plt.show()

def plot_solution_show(result, tweets, filename):
    centers = result[2]|result[4]
    radius = result[1]

    unclustered = set(tweets)
    clusters = []

    for point in centers:
        cluster = build_x_r(unclustered, radius, point, 16)
        unclustered = set(x for x in unclustered if x not in cluster)
        clusters.append(cluster)

    #print("Unclustered set", unclustered)

    #clusters.append(unclustered)
    plot(clusters, filename)


# Computes distance between two tweets.
def tweet_dist (v1, v2):
    #update dmin, dmax
    return math.sqrt((v2[2] - v1[2]) ** 2
                  + ((v2[1] - v1[1]) ** 2))


# Computes bounds on a tweet set.
# Starts from a random tweet, computes the distance to the rest of the
# tweets. Repeats from farthest/shortest tweet until it gets the same
# two tweets.

def bound (graph, comp):

    rand_nodes = [graph[i] for i in random.sample(range(len(graph)),3)]

    # Initial bound for the tweets.
    bnd = tweet_dist(rand_nodes[0], rand_nodes[1])

    # Random starting point.
    v1      = rand_nodes[2]
    v1_prev = [0.0, 0.0, 0.0]
    # Initialise vnext in case we get the bound at first try.
    vnext   = [0.0, 0.0, 0.0]
    v2      = [0.0, 0.0, 0.0]

    while v1 != v1_prev:
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            if comp(dist,bnd) and v1 != v2 and (dist>0 or bnd>0) :
                bnd   = dist
                vnext = v2
        v1_prev = v1
        v1      = vnext

    return bnd

def bound_no_zeros (graph, comp): #the above algorithm changed to ensure d_min is not zero

    rand_nodes = [graph[i] for i in random.sample(range(len(graph)),3)]

    # Initial bound for the tweets.
    bnd = tweet_dist(rand_nodes[0], rand_nodes[1])

    # Random starting point.
    v1      = rand_nodes[2]
    v1_prev = [0.0, 0.0, 0.0]
    # Initialise vnext in case we get the bound at first try.
    vnext   = [0.0, 0.0, 0.0]
    v2      = [0.0, 0.0, 0.0]

    while v1 != v1_prev:
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            if comp(dist,bnd) and v1 != v2 and (dist>0 and bnd>0) :
                bnd   = dist
                vnext = v2
        v1_prev = v1
        v1      = vnext

    return bnd

def alpha_radius_set(dmin, dmax, alpha):
    n = dmin
    if n==0:
        n = 0.1
    radii = []
    while n < dmax:
        if n >= dmin:
            radii.append(n)
        n = n * alpha
    return radii

# Finds size of a cluster within 2r distance of a given center.
def build_g_j (graph, radius, center):
    cluster = set()
    cluster.add(center)
    for node in graph:
        if tweet_dist(center, node) < 2 * radius:
            cluster.add(node)
    return len(cluster)

#Builds cluster of size val*radius for a given radius, val
def build_x_r (graph, radius, center, val):
    cluster = set()
    cluster.add(center)
    for node in graph:
        if tweet_dist(center, node) < val * radius:
            cluster.add(node)
    return cluster

#OFFLINE CLUSTERING ALGORITHM
def clustering (graph, k, radius, outlier_count):

    send_centers = set()
    centers = set()
    clusters = []
    unclustered = set(graph)

    nb_centers = 0


    while nb_centers < k:
        # loop to find max g_j
        max = 0
        remove_cluster = set()
        unclustered_temp = set(unclustered)
        while len(unclustered_temp):
            new_center = unclustered_temp.pop()
            g_j = build_g_j(unclustered, radius, new_center)
            if g_j > max:
                max_center = new_center
                max = g_j
                remove_cluster = build_x_r(unclustered, radius, new_center, 4)

        if max > 0:
            clusters.append(remove_cluster)
            centers.add(max_center)
            unclustered = set(x for x in unclustered if x not in clusters[-1])
            nb_centers += 1

    if len(unclustered) < outlier_count + 1:
        send_centers = centers

    return send_centers

#Checks if two centers are "in conflict" with one another - no support point within 2alpha*radius
def check_conflict(point, center, result, radius):
    for resu in result:
        if resu[0] == point:
            for supp in resu[1]:
                if tweet_dist(center, supp) > radius:
                    return True
        elif resu[0] == center:
            for supp in resu[1]:
                if tweet_dist(point, supp) > radius:
                    return True
    return False

#STREAMING ALGORITHM FOR CLUSTERING
def stream_clustering (graph, k, radii, outlier_count, centers, result, beta, n):

    sol=set()

    for radius in radii:

        unclustered = set(graph)

        new_center_flag=1

        while(new_center_flag > 0):
            for point in centers:
                unclustered = set(x for x in unclustered if x not in build_x_r(graph, radius, point, n))

            new_center_flag = 0

            for p in unclustered:
                x_r_alpha = build_x_r(graph, radius, p, beta)
                if len(x_r_alpha) > outlier_count:
                    centers.add(p)
                    supp = set(itertools.islice(x_r_alpha, (outlier_count+1)))
                    result.append([p, supp])
                    new_center_flag = 1
                    break

        l = len(centers)
        number_of_unclustered_points = len(unclustered)

        if number_of_unclustered_points==0 :
            break
        elif l<=k and number_of_unclustered_points < (((k-l)*outlier_count)+outlier_count):
            sol = clustering(unclustered, k-l, radius, outlier_count)
        if sol:
            break
        else:
            center_list = list(centers)
            remove = set()
            remove_supp = []
            #print("[***] Centers before conflict", centers)
            for i in range(len(center_list)):
                if center_list[i] not in remove:
                    for j in range(i + 1, len(center_list)):
                        if check_conflict(center_list[i], center_list[j], result, radius*8):
                            remove.add(center_list[j])

            centers = set(x for x in centers if x not in remove)

            if (result):
                for supp in result:
                    for old_c in remove:
                        if supp[0] == old_c:
                            remove_supp.append(supp)

                for supp in remove_supp:
                    result.remove(supp)


    return [result,radius,centers,unclustered, sol]

def insertToL(result, k, graph, outlier_count, dmax, alpha, beta, n):
    radius = result[1]
    res = result[0]
    centers = result[2]
    unclustered = result[3]

    new_unclustered = set(graph)|unclustered

    lradii = alpha_radius_set(radius, dmax, alpha)

    result_new = stream_clustering(new_unclustered, k, lradii, outlier_count, centers, res, beta, n)

    return result_new

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("[*] python3 k-center-streaming.py k outlier_count alpha beta n")
        sys.exit()

    k       = int(sys.argv[1])
    alpha = float(sys.argv[3])
    beta  = int(sys.argv[4])
    outlier_count = int(sys.argv[2])
    n = int(sys.argv[5])
    if len(sys.argv) > 7:
        stop = int(sys.argv[7])
    else:
        stop = 0

    time  = []
    start = timeit.default_timer()
    #print("[*] Parsing input file.")
    checkpoint = timeit.default_timer()

    tweets = read_tweets("dataset/twitter_10000.txt")
    tweets = tweets[:5000]

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    #print("[*] Computing dmin.")
    checkpoint = timeit.default_timer()

    dmin_bound = k + outlier_count + 1

    dmin   = bound_no_zeros(tweets[:dmin_bound], operator.lt)

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)
    #print("[*] Computing dmax.")
    checkpoint = timeit.default_timer()

    dmax   = bound(tweets[:dmin_bound], operator.gt)

    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    #print("[*] Got a dmin of:\t", dmin)
    #print("[*] Got a dmax of:\t", dmax)

    #print("[*] Initial clustering.")
    checkpoint = timeit.default_timer()
    checkpoint = timeit.default_timer()

    streambetas = alpha_radius_set(dmin, dmax, alpha)

    count = k*outlier_count

    L = stream_clustering(tweets[:(count)], k, streambetas, outlier_count, set(), [], beta, n)

    if not stop:
        for i in range(int(len(tweets)/(count))+1):
            L = insertToL(L, k, tweets[(i+1)*count:(i+2)*count], outlier_count, dmax, alpha, beta, n)


    #print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    #print("[*] Time elapsed:\t", ctime)

    filename = "figs/"+str(k)+"_"+str(outlier_count)+"_"+str(alpha)+"_"+str(beta)+"_"+str(n)+"stream.jpg"
    plot_solution_show(L, tweets, filename)

    #print("[******] Time taken", ctime)

    #print("[******] For k, epsilon, size, outliers", k, epsilon, window, outlier_count)
    print(str(k)+","+str(outlier_count)+","+str(alpha)+","+str(beta)+","+str(n)+","+str(L[1])+","+str(ctime))

    #print(time)