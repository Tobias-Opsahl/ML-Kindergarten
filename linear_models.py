"""
Some linear models implemented as classes.
This will be Perceptron, Linear Regression, Logistic regression,
One-vs-all Logistic Regression and Multinomial Logistic Regression
"""

import numpy as np


class Perceptron:
    """
    Implements the Perceptron Learning Algorithm.
    This is a simple learning algorithm that is very limited in flexibility, but
    interesting in historical and self-learning perspective.
    It will converge in a finite number of steps if the data is linearly separable
    (there exists a line or hyperplane which separates the data perfectly).
    It will classify with linear weights and a Heaviside activation function,
    for any number of features and any number of output nodes.

    The class is generelized for multiclass classification, but the target have to
    be one-hot-encoded. This means that if one runs binary classifications, the predictions
    will be a two dimensional array of size (n x 1).
    It can also train with any batch-size.

    The update rule for the weights are w[i, j] <- w[i, j] + eta * (y[j] - p[j]) * x[j].
    This means that when the prediction is correct, no update will happen.
    This is vectorized in the code.

    n: Amount of the input observations.
    m: Amount of the input features.
    k: Amount of output nodes (feature in target).

    Input: X (n x m)
    Target: Y (n x k)
    Preds: P (n x k)
    Weights: W (k x m)
    Bias: b (k x 1)
    """
    def __init__(self, n_features, n_outputs, learning_rate=0.3):
        """
        Arguments:
            n_features (int): Amount of input nodes. This should be the number of features for each observation.
            n_outputs (int): Amount of ouput nodes. This should be the number of inputs in the targets.
            learning_rate (float): The learning rate for the weight updates
        """
        self.weights = np.zeros((n_outputs, n_features))  # weights[output_node, input_node] by convention.
        self.biases = np.zeros((n_outputs, 1))
        self.eta = learning_rate

    def train(self, x_train, y_train, epochs=5, batch_size=1, verbose=True):
        """
        Trains on an input array x_train with labels y_train, according to the rule
        w[i, j] <- w[i, j] + eta * (y[j] - p[j]) * x[j] which is vectorized and batch-trained.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n x k) shaped array, representing k outputs for each of the n inputs.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            verbose (bool): If True, will print if early stopping is reached.
        """
        if len(y_train.shape) == 1:  # Expand dimensions incase of a one-dimensional (binary) target.
            y_train = np.expand_dims(y_train, 1)

        b = batch_size
        for n_epoch in range(epochs):
            prev_weights = self.weights.copy()  # Copy weights to check if the algorithm converges after each epoch.
            prev_biases = self.biases.copy()
            for i in range(int(len(x_train) / b)):
                batch = x_train[i * b: (i + 1) * b]  # Index batch
                labels = y_train[i * b: (i + 1) * b]  # Index labels
                preds = self.predict(batch)  # Do the forward pass
                # Dims: weights (k x m) + [eta (1) (labels - preds).T (k x n) @ batch (n x m)] (k x m)
                self.weights += self.eta * (labels - preds).T @ batch  # Update weights
                self.biases += self.eta * ((labels - preds).T @ np.ones((b, 1)))

            if ((i + 1) * b) < len(x_train):  # Incase batch-size mismatches with input-size, train with leftovers.
                batch = x_train[(i + 1) * b:]  # Index batch
                labels = y_train[(i + 1) * b:]
                preds = self.predict(batch)  # See the comments in the previous block
                self.weights += self.eta * (labels - preds).T @ batch
                self.biases += self.eta * ((labels - preds).T @ np.ones((len(labels), 1)))

            # Check for no updates, early stopping.
            if np.array_equal(prev_weights, self.weights) and np.array_equal(prev_biases, self.biases):
                if verbose:
                    print(f"Early stopping in Perceptron after epoch {n_epoch}")
                return

    def predict(self, x_data):
        """
        Predicts on an input array.
        The predictions are the Heaviside function of the weighted sum of the inputs.
        Vectorized the dimensions are [weights (k x m) @ x.T (m x n) + b (k x 1)] = h (k x n).

        We could also do (w @ x.T + b).T == (x @ w.T + b.T), both works the same

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.

        Returns:
            preds (np.array): (n x c) shaped array of predictions.
        """
        if len(x_data.shape) == 1:  # If single vector, turn into matrix
            x_data = np.expand_dims(x_data, 0)
        # Dimensions: [weights (k x m) @ x.T (m x n) + b (k x 1)] = h (k x n)
        h = self.weights @ x_data.T
        # The biases gets broadcasted so that its adds onto every row.
        h = h + self.biases
        return np.heaviside(h, 0).T  # Perceptron activation function


if __name__ == "__main__":
    pass
    # TODO: Make a one-hot-encoder of numpy data, that also checks for already one-hot-econding,
    # binary input or one-dimensional vector

    # # Many input features (5):
    # from common import find_accuracy
    # from sklearn.datasets import make_blobs
    # from sklearn.model_selection import train_test_split
    # np.random.seed(57)
    # n_observations = 50
    # n_features = 5
    # centers = [[0, 0, 0, 0, 0], [0, 10, 0, 0, 0], [10, 10, 0, 0, 0], [10, 0, 0, 0, 0]]
    # x_blobs, y_blobs = make_blobs(n_observations, n_features, centers=centers, cluster_std=[1, 1, 1, 1])
    # x_train_blobs, x_val_blobs, y_train_blobs, y_val_blobs = train_test_split(x_blobs, y_blobs, test_size=0.25)

    # y2 = np.zeros((y_blobs.size, y_blobs.max() + 1))
    # y2[np.arange(y_blobs.size), y_blobs] = 1

    # cl = Perceptron(n_features, 4)
    # cl.train(x_blobs, y2, batch_size=11, epochs=100)
    # preds = cl.predict(x_blobs)

    # print(f"Accuracy for Perceptron model was {find_accuracy(preds, y2)}")
