import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, K, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.K = K
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        N, D = training_data.shape

        
        
        random_indices = np.random.choice(N, self.K, replace=False)
        self.centroids = training_data[random_indices]
        print(self.centroids)

        
        for itr in range(self.max_iters):
            labels = self._assign_clusters(training_data)

            old_centroids = self.centroids.copy()

            self.centroids = self._update_centroids(training_data, labels)

            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            #  if centroid_shift <= self.tol:
            if centroid_shift <= 0.01:
                 print(f"Converged at iteration {itr}")
                 break


            if itr % 10 == 0:
                print(f"Iteration {itr}: Centroid shift = {centroid_shift}")

        pred_labels = labels

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        test_labels = self._assign_clusters(test_data)
        return test_labels
    
    def _assign_clusters(self, data):
        """
        Assigns each data point to the closest centroid.

        Arguments:
            data (np.array): Data points (N, D)

        Returns:
            labels (np.array): Labels representing the assigned clusters for each data point.
        """
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centroids(self, data, labels):
        """
        Updates the centroids by calculating the mean of the points assigned to each centroid.

        Arguments:
            data (np.array): Data points (N, D)
            labels (np.array): Cluster labels for each data point (N,)

        Returns:
            new_centroids (np.array): Updated centroids of shape (K, D)
        """
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(self.K)])
        return new_centroids
