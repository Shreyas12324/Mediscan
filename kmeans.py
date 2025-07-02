import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(data, k, max_iters=100):
    n_samples, n_features = data.shape

    # Randomly initialize the centroids
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        # Assign samples to the nearest centroids
        clusters = [[] for _ in range(k)]
        for sample in data:
            distances = [euclidean_distance(sample, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(sample)

        # Calculate new centroids
        new_centroids = np.zeros((k, n_features))
        for cluster_idx, cluster in enumerate(clusters):
            new_centroids[cluster_idx] = np.mean(cluster, axis=0)

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# Movie rating data
data = np.array([
    [2, 4, 3, 3, 2, 3, 3],
    [1, 4, 3, 2, 1, 5, 3],
    [4, 3, 3, 3, 4, 1, 5],
    [1, 2, 4, 3, 1, 2, 1],
    [0, 3, 5, 5, 0, 2, 2],
    [2, 3, 1, 1, 2, 2, 2],
    [5, 5, 2, 2, 5, 4, 4],
    [2, 1, 0, 2, 2, 3, 3],
    [3, 2, 2, 2, 3, 4, 2],
    [3, 2, 1, 4, 4, 5, 1],
    [3, 4, 4, 3, 5, 1, 4],
    [4, 3, 1, 4, 1, 2, 1],
    [5, 5, 2, 3, 2, 0, 0],
    [1, 1, 4, 3, 0, 2, 2],
    [2, 2, 3, 3, 2, 1, 5],
    [0, 0, 4, 4, 1, 4, 2],
    [2, 0, 3, 5, 4, 3, 4],
    [1, 2, 3, 1, 2, 3, 4],
    [4, 5, 3, 2, 4, 5, 3],
    [1, 2, 4, 0, 3, 1, 2]
])

# Perform K-Means clustering
k = 2
centroids, clusters = kmeans(data, k)

# Display the clusters
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
