import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.spatial import distance
import numpy as np
from sklearn.datasets import make_blobs
import addcopyfighandler

'''Hard Clustering K-Means Algorithm Implementation'''
def converged(centroid_prev, centroid_curr, disp):
    if(centroid_prev == None):
        return False
    total_disp = 0
    for idx in range(len(centroid_prev)):
        prev = np.asarray(centroid_prev[idx]) #2 element arr
        curr = np.asarray(centroid_curr[idx]) #2 element arr
        total_disp+=np.sum(np.absolute(prev-curr))
    print("Total Displacement is: ", total_disp)
    return total_disp <= disp
'''Output Centroids of clustered dataset'''
def k_means(dataset, k):
    X = np.asarray(dataset["X"])
    Y = np.asarray(dataset["Y"])
    # plt.scatter(X, Y, marker='.')
    # plt.title("Before Clustering")
    # plt.show()
    k_buckets = [[] for x in range(0,k)]
    max_x = np.amax(X)
    max_y = np.amax(Y)
    # [np.random.randint(low=0, high=max_x), np.random.randint(low=0, high=max_y)]
    #centroids = [[np.random.randint(low=0, high=max_x), np.random.randint(low=0, high=max_y)] for i in range(0, k)]
    centroids = [[i * (max_x/(k+1)), max_y/2] for i in range(0, k)]
    print("Centroids: ", centroids)
    # print("Converged? : ", converged(centroids, centroids, 0))
    percent_disp = .01 # Convergence happens if centroids moved within 5 percent of window frame
    disp = percent_disp * max_x
    print("Max Disp is: ", disp)
    centroid_prev = None 
    max_iter = 50
    iter = 0
    bucket_copy = None
    while(not converged(centroid_prev, centroids, disp)):
        #Fill in buckets 
        for x,y in list(zip(X,Y)):
            min_dist = 1000000
            min_pos = -1
            for i in range(len(centroids)):
                #dst = distance.euclidean([x,y], centroids[i])
                dst = np.linalg.norm(np.asarray([x,y]) - np.asarray(centroids[i]))
                if(dst < min_dist):
                    min_dist = dst 
                    min_pos = i 
            k_buckets[min_pos].append([x,y])

        #Re-write each centroid as the mean of the bucket
        bucket_copy = k_buckets.copy()
        centroid_prev = centroids.copy()
        for idx in range(len(k_buckets)):
            bucket = k_buckets[idx]
            centroids[idx] = np.mean(bucket, axis=0)
            k_buckets[idx] = []
        iter+=1
    return centroids, bucket_copy

np.random.seed(0)
X, y = make_blobs(n_samples=500, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
data_x = X[:, 0]
data_y = X[:, 1]
dataset = {"X": data_x, "Y": data_y}
cents, buckets = k_means(dataset=dataset, k=4)

print("Centroids: ", cents)

cents_X = [cents[x][0] for x in range(len(cents))]
cents_Y = [cents[x][1] for x in range(len(cents))]
print("cents_X: ", cents_X)
print("cents_Y: ", cents_Y)
plt.scatter(X[:, 0], X[:, 1], marker='.')


colors = cm.rainbow(np.linspace(0, 1, len(cents)))

#Plotting Centroids
for idx in range(len(cents)):
    x = [cents_X[idx]]
    y = [cents_Y[idx]]
    color = colors[idx]
    plt.scatter(x, y, color=color)

#Plotting Data Points
for idx in range(len(buckets)):
    bucket = buckets[idx]
    x = []
    y = []
    for point in bucket:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y, color=colors[idx])
plt.savefig('K_Means_Clustering_Walid_Implementation.png')
plt.show()
# bucket = [[10,4], [2,3], [4,5]]
# mean = np.mean(bucket, axis = 0)
# print("Mean: ", mean, "of length: ", len(mean))