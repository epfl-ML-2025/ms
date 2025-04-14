import numpy as np
import itertools




class KMeans(object):
    """
    kNN classifier object.
    """

    def init_centers(data, K):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """
        ### WRITE YOUR CODE HERE
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:K]
        # Use these index to select centers from data
        centers = data[random_idx[:K]]
        
        return centers
    
    def compute_distance(data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]

        distances = np.zeros((N, K))
        for k in range(K):
            # Compute the euclidean distance for each data to each center
            center = centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
            
        return distances
    
    def find_closest_cluster(distances):
        """
        Assign datapoints to the closest clusters.
        
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        ### WRITE YOUR CODE HERE
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments
    
    def compute_centers(data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        ### WRITE YOUR CODE HERE
        N = data.shape[0]
        D = data.shape[1]

        centers = np.zeros((K,D))

        for i in range(K):
            cluster = data[cluster_assignments == i]
            Nk = cluster.shape[0]
            if (Nk > 0):
                centers[i] = np.sum(cluster, axis = 0) / Nk
        return centers
    
    
    def k_means(data, K, max_iter):
        """
        Main function that combines all the former functions together to build the K-means algorithm.
        
        Arguments: 
            data: array of shape (N, D) where N is the number of data samples, D is number of features.
            K: int, the number of clusters.
            max_iter: int, the maximum number of iterations
        Returns:
            centers: array of shape (K, D), the final cluster centers.
            cluster_assignments: array of shape (N,) final cluster assignment for each data point.
        """
        # Initialize the centers
        centers = KMeans.init_centers(data, K)

        # Loop over the iterations
        for i in range(max_iter):
            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}...")
            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            ### WRITE YOUR CODE HERE

            distances = KMeans.compute_distance(data, centers)
            closest_cluster = KMeans.find_closest_cluster(distances)
            centers = KMeans.compute_centers(data, closest_cluster, K)

            # End of the algorithm if the centers have not moved
            if (np.all(old_centers == centers)):  ### WRITE YOUR CODE HERE
                print(f"K-Means has converged after {i+1} iterations!")
                break
        
        # Compute the final cluster assignments
        ### WRITE YOUR CODE HERE
        distances = KMeans.compute_distance(data, centers)
        closest_cluster = KMeans.find_closest_cluster(distances)
        centers = KMeans.compute_centers(data, closest_cluster, K)

        cluster_assignments = closest_cluster
        
        return centers, cluster_assignments
    
    def assign_labels_to_centers(centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        ### WRITE YOUR CODE HERE
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(true_labels[cluster_assignments == i].astype(int)))
            cluster_center_label[i] = label
        return cluster_center_label

    def predict_with_centers(data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments: 
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns: 
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        ### WRITE YOUR CODE HERE
        # Compute cluster assignments
        distances = KMeans.compute_distance(data, centers)
        cluster_assignments = KMeans.find_closest_cluster(distances)

        # Convert cluster index to label
        new_labels = cluster_center_label[cluster_assignments]
        return new_labels


    def __init__(self, K=5, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.K = K
        self.max_iters = max_iters
        self.centers = None
        self.cluster_center_labels = None

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

        centers, cluster_assignments = KMeans.k_means(training_data, self.K, self.max_iters)
        self.centers = centers
        
        # Assigner un label à chaque centre par vote majoritaire
        self.cluster_center_labels = KMeans.assign_labels_to_centers(centers, cluster_assignments, training_labels)
        
        # Prédire les labels sur les données d'entraînement
        predicted_labels = KMeans.predict_with_centers(training_data, centers, self.cluster_center_labels)
        return predicted_labels

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
        test_labels = KMeans.predict_with_centers(test_data, self.centers, self.cluster_center_labels)
        
        return test_labels
    

