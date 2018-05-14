import numpy as np


class Perceptron(object):
    """
    perceptron classifier.

    Parameters
    alpha: learning rate (float)
    no_epochs: number of epochs (int)

    Attributes
    weights_: weights (1d array)
    errors_: number of misclassification (list)
    """

    def __init__(self, alpha=0.01, no_epochs=10):
        self.alpha = alpha
        self.no_epochs = no_epochs

    def fit(self, X, y):
        """
        Fit training data

        :param X: (array-like), shape = [no_samples, no_features]
        :param y: (array-like), shape = [no_samples]
                    target values
        :return:
        self: object
        """

        self.weights_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.no_epochs):
            errors = 0
            for xi, target in zip(X,y):
                update = self.alpha * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        calculate input
        :param X: input
        :return: weigthed summation of input
        """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """

        :param X: input
        :return: class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
