import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        training_labels_one_hot = label_to_onehot(training_labels)
        self.W = np.random.normal(0, 0.1, (D, C))
        for itr in range(self.max_iters):
            # First compute W diff:
            deltaW = self.gradient_logistic_multi(training_data, training_labels_one_hot, self.W)
            # Second update W:
            self.W = self.W - self.lr*deltaW
            if itr % 5 == 0:
                print('loss at iteration', itr, ":", self.loss_logistic_multi(training_data, training_labels_one_hot, self.W))
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        pred_labels = np.argmax(self.f_softmax(test_data, self.W), axis=1)
        return pred_labels
    def f_softmax(self,data, W):
        """
        Softmax function for multi-class logistic regression.
    
        """
        logits = data @ W 
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits_stable)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities
    def loss_logistic_multi(self,data, labels, w):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        """
        softmax_probs = self.f_softmax(data, w) 
        loss = -np.sum(labels * np.log(softmax_probs))  

        return loss
    def gradient_logistic_multi(self,data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        """
        prob = self.f_softmax(data, W)
        return np.dot(data.T, (prob - labels))