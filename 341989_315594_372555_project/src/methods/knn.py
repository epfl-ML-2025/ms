import numpy as np


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def euclidean_dist(self, example, training_data):
        """
        Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD)
        Outputs:
            euclidean distances: shape (N,)
        """
        distance = np.sqrt(np.sum((training_data - example)**2, axis=1))
        return distance

    def find_k_nearest_neighbors(self, distances):
        """
        Find the indices of the k smallest distances from a list of distances.
        Tip: use np.argsort()

        Inputs:
            distances: shape (N,)
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)[:self.k]
        return indices

    def predict_label(self, neighbor_labels):
        """
        Return the most frequent label in the neighbors.
        In case of a tie, return the label of the closest neighbor among the tied labels.

        Inputs:
            neighbor_labels: shape (N,)
        Outputs:
            most frequent label
        """
        counts = np.bincount(neighbor_labels.astype(int))
        max_count = np.max(counts)
        best_labels = np.where(counts == max_count)[0]

        if len(best_labels) == 1:
            return best_labels[0]
        else:
            for label in neighbor_labels:
                if label in best_labels:
                    return label

    def kNN_one_example(self, unlabeled_example, training_data, training_labels):
        """
        Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,)
            training_features: shape (NxD)
            training_labels: shape (N,)
        Outputs:
            predicted label
        """
        
        distances = self.euclidean_dist(unlabeled_example, training_data)
        
        nn_indices = self.find_k_nearest_neighbors(distances)
        
        neighbor_labels = training_labels[nn_indices]

        best_label = self.predict_label(neighbor_labels)

        return best_label

    def kNN(self, unlabeled, training_data, training_labels):
        """
        Return the labels vector for all unlabeled datapoints.

        Inputs:
            unlabeled: shape (MxD)
            training_features: shape (NxD)
            training_labels: shape (N,)
        Outputs:
            predicted labels: shape (M,)
        """
        return np.apply_along_axis(
            func1d=self.kNN_one_example,
            axis=1,
            arr=unlabeled,
            training_data=training_data,
            training_labels=training_labels
        )

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        # Store training data and labels as class attributes to be able to use them in the predict method
        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.kNN(training_data, training_data, training_labels)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = self.kNN(test_data, self.training_data, self.training_labels)
        return test_labels
