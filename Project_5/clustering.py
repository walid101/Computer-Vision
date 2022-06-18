import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.spatial import distance
import numpy as np
from sklearn.datasets import make_blobs, load_diabetes
import addcopyfighandler


def find_outliers(cents, buckets, num_std):
    outliers = []
    for i in range(0, len(cents)):
        centroid = cents[i]
        #print("Centroid is at: ", centroid)
        bucket = buckets[i]
        std = np.std(bucket)
        #print("Standard Deviation at bucket ", str(i), " is ", std)
        for point in bucket:
            point_dist = np.linalg.norm(np.asarray(point) - np.asarray(centroid))
            #print("Point : ", point)
            if((point_dist / std) > 3.5):
                #print("OUTLIER DETECTED! With Centroid: ", centroid, " Outlier: ", point, " Distance: ", point_dist, " standard dev: ", std)
                outliers.append(point) 
    return outliers

'''Hard Clustering K-Means Algorithm Implementation'''
def converged(centroid_prev, centroid_curr, disp):
    if(centroid_prev == None):
        return False
    total_disp = 0
    for idx in range(len(centroid_prev)):
        prev = np.asarray(centroid_prev[idx]) #2 element arr
        curr = np.asarray(centroid_curr[idx]) #2 element arr
        total_disp+=np.sum(np.absolute(prev-curr))
    ## print("Total Displacement is: ", total_disp)
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
    ## print("Centroids: ", centroids)
    # print("Converged? : ", converged(centroids, centroids, 0))
    percent_disp = .01 # Convergence happens if centroids moved within 5 percent of window frame
    disp = percent_disp * max_x
    ## print("Max Disp is: ", disp)
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

def k_medians(dataset, k):
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
    ## print("Centroids: ", centroids)
    # print("Converged? : ", converged(centroids, centroids, 0))
    percent_disp = .01 # Convergence happens if centroids moved within 5 percent of window frame
    disp = percent_disp * max_x
    ## print("Max Disp is: ", disp)
    centroid_prev = None 
    max_iter = 50
    iter = 0
    bucket_copy = None
    while(not converged(centroid_prev, centroids, disp)):
        #Fill in buckets 
        try:
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
                centroids[idx] = np.median(bucket, axis=0)
                k_buckets[idx] = []
        except:
            continue
        iter+=1
    return centroids, bucket_copy

np.random.seed(0)
#centers=[[4,4], [-2, -1], [2, -3], [1, 1]]
X, y = make_blobs(n_samples=500,centers=[[1, 1]], cluster_std=0.9)
data_x = X[:, 0]
data_y = X[:, 1]
num_outliers = 10
for i in range(num_outliers):
    data_x = np.append(data_x, np.random.randint(0,20))
    data_y = np.append(data_y, np.random.randint(0,20))
print(data_x)
dataset = {"X": data_x, "Y": data_y}
cents, buckets = k_medians(dataset=dataset, k=3)
#cents_b, buckets_b = k_medians(dataset=dataset, k=4)

# print("Centroids: ", cents)
cents_X = [cents[x][0] for x in range(len(cents))]
cents_Y = [cents[x][1] for x in range(len(cents))]

# cents_b_X = [cents_b[x][0] for x in range(len(cents_b))]
# cents_b_Y = [cents_b[x][1] for x in range(len(cents_b))]
# print("cents_X: ", cents_X)
# print("cents_Y: ", cents_Y)

plt.scatter(X[:, 0], X[:, 1], marker='.')


colors = cm.rainbow(np.linspace(0, 1, len(cents)))

#Plotting Data Points
for idx in range(len(buckets)):
    bucket = buckets[idx]
    x = []
    y = []
    for point in bucket:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y, color=colors[idx])

#Plotting Centroids
for idx in range(len(cents)):
    x = [cents_X[idx]]
    y = [cents_Y[idx]]
    color = colors[idx-1]
    plt.scatter(x, y, color=color)

outliers = find_outliers(cents, buckets, 3.5)
print("Number of outliers: ", len(outliers))
print("Outliers: ", outliers)

#Plotting Outliers
for idx in range(len(outliers)):
    x = [outliers[idx][0]]
    #print("x : ", x)
    y = [outliers[idx][1]]
    #print("y : ", y)
    plt.scatter(x, y, color='black')


plt.savefig('K_Means_Clustering_Walid_Implementation.png')
plt.show()
