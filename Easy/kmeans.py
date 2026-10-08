# 1. Initialize k centroids
# 2. Repeat:
#    a. ASSIGN each point to nearest centroid
#    b. UPDATE each centroid to mean of its assigned points
#    c. CHECK CONVERGENCE — if centroids barely moved, stop

import numpy as np 


def euclidean_dist(X, centeroids):
    """
    X : [num_samples,features]
    centeroids : [k,features]
  return [num_samples,k] matrix of dist    
    """
    diff = X[:,np.newaxis,:] - centeroids[np.newaxis,:,:]
     # [N,D] --- [N,1,D] - [k,D] --- [1,K,D] == [N,K,D]
    return np.sqrt((diff**2).sum(axis=2))

def kmeans(X,k,max_iters=100,tol=1e-4):

    n = len(X)

    idx = np.random.choice(n, k, replace=False)
    centroids  = X[idx].copy() 

    for _ in range(max_iters):

        distances = np.linalg.norm(X[:,None]-centroids[None,:],axis=2) # (n,k)
        labels = np.argmin(distances,axis=1) # (n,)

        new_centroids = np.array([
            X[labels==j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)
        ])

# • labels == j: This creates a boolean mask (a list of True and False values) identifying which data points in X belong to cluster j.
# • X[labels == j]: This uses that mask to filter X, extracting only the data points assigned to cluster j.
# • .mean(axis=0): This calculates the average position of those filtered points. Specifying axis=0 ensures that the average is calculated column-by-column (feature-by-feature), returning a single coordinate point representing the new center of the cluster.

        if np.linalg.norm(new_centroids - centroids) < tol :
            break

        centroids = new_centroids



    return centroids, labels




