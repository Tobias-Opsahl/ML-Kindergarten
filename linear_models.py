"""
Some linear models implemented as classes.
This will be Perceptron, Linear Regression, Logistic regression,
One-vs-all Logistic Regression and Multinomial Logistic Regression
"""

import numpy as np
from common import (find_mse, find_accuracy, find_binary_cross_entropy, find_multiclass_cross_entropy,
                    check_array, check_arrays, integer_one_hot_encode, sigmoid, softmax)


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

    Methods:
        train: Trains the perceptron.
        predict: Predict on data.
    """
    def __init__(self, n_features=None, n_outputs=None, learning_rate=0.3):
        """
        Initializes weights and biases. If n_features or n_outputs is None, the weights and biases will be set
        in training, depending on the sizes of the x_train and y_train arrays.

        Arguments:
            n_features (int): Amount of input nodes. This should be the number of features for each observation.
            n_outputs (int): Amount of ouput nodes. This should be the number of inputs in the targets.
            learning_rate (float): The learning rate for the weight updates
        """
        self.weights = None
        self.biases = None
        self.eta = learning_rate
        if n_features is not None and n_outputs is not None:  # Initialize weights and biases
            self.weights = np.zeros((n_outputs, n_features))  # weights[output_node, input_node] by convention.
            self.biases = np.zeros((n_outputs, 1))

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
        check_array(x_train, check_shape=2)  # Check that x_train has 2 dimensions.
        check_arrays(x_train, y_train, dims=[0])  # Check that the amount of (n) points are equal.

        if len(y_train.shape) == 1:  # 1d array, either one-hot-encode or expand-dims.
            if np.unique(y_train).shape[0] > 2:  # More than two classes (multiclass), so we one-hot-encode
                y_train = integer_one_hot_encode(y_train)
            else:
                y_train = np.expand_dims(y_train, 1)  # Expand dimensions of a one-dimensional (binary) target.

        if self.weights is None:  # Initialize weights if it has not yet been done.
            n_features = x_train.shape[1]
            n_outputs = y_train.shape[1]
            self.weights = np.zeros((n_outputs, n_features))  # weights[output_node, input_node] by convention.
            self.biases = np.zeros((n_outputs, 1))

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


class LinearRegression:
    """
    Class for implementing linear regression.
    Can be solved both with exact analytical solution, gradient descent and newtons method.

    Bias vector is added into the x-matrix and included in the weight matrix if with_bias is True.

    Methods:
        train: Trains the linear regression, either analytically, with SGD or newtons method.
        predict: Predict on data.
    """
    def __init__(self, n_features=None, with_bias=True, verbose=False):
        """
        Initializes betas parameters according to the amount of features in the training-data.
        If not provided, will be set in the training.

        Arguments:
            n_features (int): The amount of features in the training data (without the bias).
            with_bias (bool): Weather or not to add a bias (intercept) to the model.
            verbose (bool): If True, will print some intermediate output from gradient descent.
        """
        self.betas = None
        self.with_bias = with_bias
        self.verbose = verbose
        if n_features is not None:
            self.n_features = n_features
            self.betas = np.zeros(self.n_features + int(with_bias))  # Initialize parameters plus optinal bias

    def train(self, x_train, y_train, method="analytic", eval_set=None,
              eta=0.01, epochs=20, batch_size=1, tol=None, tol_rounds=5):
        """
        Trains model.
        This can be done analytically, with vanilla stochastic gradient descent (SGD) or with newtons method.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            method (str): Determines the method that will be used to find the betas.
                Must be in ["analytic", "gradient_descent", "newton"].

            The following parameters are only used for "gradient_descent" and "newtwon":
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        check_arrays(x_train, y_train, dims=[0])  # Check that the amount of (n) points are equal.

        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_train = np.expand_dims(x_train, 1)

        if self.betas is None:  # Initialize beta-parameters
            self.n_features = x_train.shape[1]
            self.betas = np.zeros(self.n_features + int(self.with_bias))  # Initialize parameters plus optinal bias

        if self.with_bias:
            x_train = self._add_bias(x_train)

        if method == "analytic":
            self._train_analytic(x_train, y_train)
        elif method == "gradient_descent" or method.upper() == "SGD":
            self._train_gradient_descent(x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds)
        elif method == "newton":
            self._train_newton(x_train, y_train, eval_set, epochs, tol, tol_rounds)
        else:
            message = "Argument \"method\" to train must be in [\"analytic\", \"gradient_descent\", \"newton\"]. "
            message += f"Was {method}."
            raise ValueError(message)

    def predict(self, x_data, have_bias=False):
        """
        Predicts on data.
        Prediction for a datapoint is beta * x.

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        if len(x_data.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_data = np.expand_dims(x_data, 1)

        if not have_bias and self.with_bias:  # Add bias
            x_data = self._add_bias(x_data)
        return self.betas @ x_data.T  # Compute the weighted sum for each datapoint.

    def _train_analytic(self, x_train, y_train):
        """
        Trains the model by solving the linear equations that solves for lowest MSE for the beta parameters.
        This is done naively with matrix multiplications and inversions, and without any optimizations.
        The formula for the bets parameters are (X.T @ X)^(-1) X.T @ y

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
        """
        # (x.T [m x n] @ x [n x m])^(-1) [m x m] @ x.T [m x n] @ y [n]
        self.betas = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

    def _train_gradient_descent(self, x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds):
        """
        Trains data with vanilla SGD.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        b = batch_size
        n = len(x_train)
        shuffle_indices = np.random.choice(n, n, replace=False)  # Shuffle training data
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        self.train_losses = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)  # Will only be filled in if eval_set is not None.
        for n_epoch in range(epochs):
            for i in range(int(n / b)):  # Loop over all the minibatches for SGD
                upper_index = np.min((n, (i + 1) * b))  # Round of in case batch-size does not match up with n.
                n_data = upper_index - i * b  # Should be b unless last iteration
                batch = x_train[i * b: upper_index]  # Index batch
                targets = y_train[i * b: upper_index]  # Index labels
                preds = self.predict(batch, have_bias=True)  # Do the forward pass
                # Dims: [eta / n_data (1) (targets - preds) (n) @ batch (n x m)] (m)
                self.betas += (eta / n_data) * (targets - preds) @ batch  # Update weights
                self.train_losses[n_epoch] += find_mse(preds, targets) * n_data / n

            self._update_val_loss(eval_set, n_epoch)  # Update eval set
            stop_training = self._check_tolerance(tol, tol_rounds, n_epoch)  # Check for criteria
            if stop_training:  # We have reached our desired tolerance, quit training
                return

        self.epochs_ran = n_epoch + 1  # Save the amount of epochs ran, for plotting

    def _train_newton(self, x_train, y_train, eval_set, epochs, tol, tol_rounds):
        """
        Trains data with the use of second derivative.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            epochs (int): The amount of time we loop through the training data.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        n = x_train.shape[0]
        self.train_losses = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)  # Will only be filled in if eval_set is not None.
        for n_epoch in range(epochs):
            grad = self._loss_grad(x_train, y_train)
            hessian = self._loss_hessian(x_train)
            self.betas += np.linalg.inv(hessian) @ grad
            preds = self.predict(x_train, have_bias=True)
            self.train_losses[n_epoch] += find_mse(preds, y_train)

            self._update_val_loss(eval_set, n_epoch)  # Update eval set
            stop_training = self._check_tolerance(tol, tol_rounds, n_epoch)  # Check for criteria
            if stop_training:  # We have reached our desired tolerance, quit training
                return

        self.epochs_ran = n_epoch + 1

    def _loss_grad(self, x_data, y_data):
        """
        Gradient of the loss function, mse.
        d (1 / 2N sum_i(y - pred)^2)/d beta_j = - 1/N sum_i(y - pred) x_[i, j]
        Vectorized: -1 / N (y - pred) [n] @ X [n x m]: [m]

        Arguments:
            x_data (np.array): (n x m) array of input to calculate loss on.
            y_data (np.array): (n) rray of true targets.

        Returns:
            grad (np.array): (n) array of the gradient with respect to mse.
        """
        preds = self.predict(x_data, have_bias=True)
        diff = (y_data - preds)
        return diff @ x_data / x_data.shape[0]

    def _loss_hessian(self, x_data):
        """
        Hessian of the loss function, mse.
        d^2 (1 / 2N sum_i(y - pred)^2)d beta_j d beta_k = 1 / N sum_i x_[i, j] x_[i, k] = 1 / N x[:, j] @ x[:, k]
        Vectorized: 1 / N X.T [m x n] @ X [n x m]: [m x m]

        Arguments:
            x_data (np.array): (n x m) array of input to calculate loss on.
            y_data (np.array): (n) rray of true targets.

        Returns:
            hessian (np.array): (m x m) array of the hessian with respect to mse.
        """
        hessian = (x_data.T @ x_data) / x_data.shape[0]
        return hessian

    def _update_val_loss(self, eval_set, n_epoch):
        """
        Updates eval-loss on eval-set, if it is not None.

        Arguments:
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                    eval_set = (x_val, y_val) of shapes [k x m] and [k].
            n_epoch (int): The epoch number we are currently on.
        """
        if eval_set is not None:  # Update eval losses
            x_val, y_val = eval_set
            preds = self.predict(x_val)
            self.val_losses[n_epoch] = find_mse(preds, y_val)

    def _check_tolerance(self, tol, tol_rounds, n_epoch):
        """
        Check if we have reached our desired tolerance, and if we have, quit the training.

        Arguments:
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
            n_epoch (int): The epoch number we are currently on.

        Returns:
            stop_training (bool): A boolean representing wether or not we have reached our desired tolerance.
        """
        if tol is not None and n_epoch > tol_rounds:  # Check if we have not improved above the given tolerance
            best_prev = self.train_losses[:(n_epoch - tol_rounds)].min()  # Best loss until tol_rounds epcohs ago
            best_recent = self.train_losses[(n_epoch - tol_rounds):n_epoch].min()  # Best loss the last tol_rounds
            if (best_prev - best_recent) < tol:  # We have not improved enough
                if self.verbose:
                    print(f"Early stopping after epoch {n_epoch + 1} with tolerance {tol}.")
                self.epochs_ran = n_epoch + 1
                return True
        else:
            return False

    def _add_bias(self, x_data):
        """
        Adds a bias to the data. The bias will be in the first (zero'th) column, and consist of ones.

        Arguments:
            x_data (np.array): Array of data to add the bias on

        Returns:
            x_data (np.array): The same data, but with a one-column in the zero'th column index.
        """
        return np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)


class LogisticRegression:
    """
    Class for implementing logistic regression.
    Can be solved both with gradient descent or newtons method.

    Bias vector is added into the x-matrix and included in the weight matrix if with_bias is True.

    Methods:
        train: Trains the linear regression with SGD or newtons method.
        predict_probabilities: Predict probabilities on data (sigmoid values).
        predict: Predict on data (0 or 1).
    """
    def __init__(self, n_features=None, with_bias=True, verbose=False):
        """
        Initializes betas parameters according to the amount of features in the training-data.
        If not provided, will be set in the training.

        Arguments:
            n_features (int): The amount of features in the training data (without the bias).
            with_bias (bool): Weather or not to add a bias (intercept) to the model.
            verbose (bool): If True, will print some intermediate output from gradient descent.
        """
        self.betas = None
        self.with_bias = with_bias
        self.verbose = verbose
        if n_features is not None:
            self.n_features = n_features
            self.betas = np.zeros(self.n_features + int(with_bias))  # Initialize parameters plus optinal bias

    def train(self, x_train, y_train, method="gradient_descent", eval_set=None,
              eta=0.01, epochs=20, batch_size=1, tol=None, tol_rounds=5):
        """
        Trains model.
        This can be done with stochastic gradient descent (SGD) or with newtons method.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            method (str): Determines the method that will be used to find the betas.
                Must be in ["gradient_descent", "newton"].
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        check_arrays(x_train, y_train, dims=[0])  # Check that the amount of (n) points are equal.

        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_train = np.expand_dims(x_train, 1)

        if self.betas is None:  # Initialize beta-parameters
            self.n_features = x_train.shape[1]
            self.betas = np.zeros(self.n_features + int(self.with_bias))  # Initialize parameters plus optinal bias

        if self.with_bias:
            x_train = self._add_bias(x_train)

        if method == "gradient_descent" or method.upper() == "SGD":
            self._train_gradient_descent(x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds)
        elif method == "newton":
            self._train_newton(x_train, y_train, eval_set, epochs, tol, tol_rounds)
        else:
            message = "Argument \"method\" to train must be in [\"gradient_descent\", \"newton\"]. "
            message += f"Was {method}."
            raise ValueError(message)

    def predict_probabilities(self, x_data, have_bias=False):
        """
        Predicts on data.
        Prediction for a datapoint is sigmoid(beta * x).

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        if len(x_data.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_data = np.expand_dims(x_data, 1)

        if not have_bias and self.with_bias:  # Add bias
            x_data = self._add_bias(x_data)
        z = self.betas @ x_data.T  # Compute the weighted sum for each datapoint.
        return sigmoid(z)

    def predict(self, x_data, have_bias=False, threshold=0.5):
        """
        Predicts on data, 0 or 1 for each class.
        Prediction for a datapoint is 1[sigmoid(beta * x) > threshold].

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.
            threshold (float): Threshold to determine if a probability will predict 0 or 1.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        probs = self.predict_probabilities(x_data, have_bias)
        preds = (probs > threshold).astype(int)
        return preds

    def _train_gradient_descent(self, x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds):
        """
        Trains data with vanilla SGD.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        b = batch_size
        n = len(x_train)
        shuffle_indices = np.random.choice(n, n, replace=False)  # Shuffle training data
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        self.train_losses = np.zeros(epochs)  # Store training stats
        self.train_accuracies = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)  # Will only be filled in if eval_set is not None.
        self.val_accuracies = np.zeros(epochs)

        for n_epoch in range(epochs):
            for i in range(int(n / b)):  # Loop over all the minibatches for SGD
                upper_index = np.min((n, (i + 1) * b))  # Round of in case batch-size does not match up with n.
                n_data = upper_index - i * b  # Should be b unless last iteration
                batch = x_train[i * b: upper_index]  # Index batch
                targets = y_train[i * b: upper_index]  # Index labels
                pred_probs = self.predict_probabilities(batch, have_bias=True)  # Do the forward pass
                preds = self.predict(batch, have_bias=True)  # Do the forward pass
                # Dims: [eta / n_data (1) (targets - preds) (n) @ batch (n x m)] (m)
                self.betas += (eta / n_data) * (targets - pred_probs) @ batch  # Update weights
                self.train_losses[n_epoch] += find_binary_cross_entropy(pred_probs, targets) * n_data / n
                self.train_accuracies[n_epoch] += find_accuracy(preds, targets) * n_data / n

            self._update_val_loss(eval_set, n_epoch)  # Update eval set
            stop_training = self._check_tolerance(tol, tol_rounds, n_epoch)  # Check for criteria
            if stop_training:  # We have reached our desired tolerance, quit training
                return

        self.epochs_ran = n_epoch + 1  # Save the amount of epochs ran, for plotting

    def _train_newton(self, x_train, y_train, eval_set, epochs, tol, tol_rounds):
        """
        Trains data with the use of second derivative.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            epochs (int): The amount of time we loop through the training data.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        n = x_train.shape[0]
        self.train_losses = np.zeros(epochs)  # Initalize losses over epochs
        self.train_accuracies = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)  # Will only be filled in if eval_set is not None.
        self.val_accuracies = np.zeros(epochs)
        for n_epoch in range(epochs):
            grad = self._loss_grad(x_train, y_train)
            hessian = self._loss_hessian(x_train)
            self.betas += np.linalg.inv(hessian) @ grad
            pred_probs = self.predict_probabilities(x_train, have_bias=True)
            preds = self.predict(x_train, have_bias=True)
            self.train_losses[n_epoch] += find_binary_cross_entropy(pred_probs, y_train)
            self.train_accuracies[n_epoch] += find_accuracy(preds, y_train)

            self._update_val_loss(eval_set, n_epoch)  # Update eval set
            stop_training = self._check_tolerance(tol, tol_rounds, n_epoch)  # Check for criteria
            if stop_training:  # We have reached our desired tolerance, quit training
                return

        self.epochs_ran = n_epoch + 1

    def _loss_grad(self, x_data, y_data):
        """
        Gradient of the loss function, binary cross entropy of sigmoid.
        d (-1 / N sum_i (y_i log(pred_i) + (1 - y_i) log(1 - pred_i)) )/d beta_j = - 1 / N sum_i (y_i - pred_i) x_[i, j]
        Vectorized: 1 / N (y - pred) [n] @ X [n x m]: [m]

        Arguments:
            x_data (np.array): (n x m) array of input to calculate loss on.
            y_data (np.array): (n) rray of true targets.

        Returns:
            grad (np.array): (n) array of the gradient with respect to loss.
        """
        preds = self.predict(x_data, have_bias=True)
        diff = (y_data - preds)
        return diff @ x_data / x_data.shape[0]

    def _loss_hessian(self, x_data):
        """
        Hessian of the loss function, mse.
        d (- 1/N sum_i (y_i log(pred_i) + (1 - y_i) log(1 - pred_i)) ) / d beta_j
            = 1 / N sum_k x_[k, j] x_[k, i] = 1 / N x[:, i] @ x[:, j]
        Vectorized: 1 / N X.T [m x n] @ X [n x m]: [m x m]

        Arguments:
            x_data (np.array): (n x m) array of input to calculate loss on.
            y_data (np.array): (n) rray of true targets.

        Returns:
            hessian (np.array): (m x m) array of the hessian with respect to loss.
        """
        hessian = (x_data.T @ x_data) / x_data.shape[0]
        return hessian

    def _update_val_loss(self, eval_set, n_epoch):
        """
        Updates eval-loss on eval-set, if it is not None.

        Arguments:
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                    eval_set = (x_val, y_val) of shapes [k x m] and [k].
            n_epoch (int): The epoch number we are currently on.
        """
        if eval_set is not None:  # Update eval losses
            x_val, y_val = eval_set
            pred_probs = self.predict_probabilities(x_val)
            preds = self.predict(x_val)
            self.val_losses[n_epoch] = find_binary_cross_entropy(pred_probs, y_val)
            self.val_accuracies[n_epoch] = find_accuracy(preds, y_val)

    def _check_tolerance(self, tol, tol_rounds, n_epoch):
        """
        Check if we have reached our desired tolerance, and if we have, quit the training.

        Arguments:
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
            n_epoch (int): The epoch number we are currently on.

        Returns:
            stop_training (bool): A boolean representing wether or not we have reached our desired tolerance.
        """
        if tol is not None and n_epoch > tol_rounds:  # Check if we have not improved above the given tolerance
            best_prev = self.train_losses[:(n_epoch - tol_rounds)].min()  # Best loss until tol_rounds epcohs ago
            best_recent = self.train_losses[(n_epoch - tol_rounds):n_epoch].min()  # Best loss the last tol_rounds
            if (best_prev - best_recent) < tol:  # We have not improved enough
                if self.verbose:
                    print(f"Early stopping after epoch {n_epoch + 1} with tolerance {tol}.")
                self.epochs_ran = n_epoch + 1
                return True
        else:
            return False

    def _add_bias(self, x_data):
        """
        Adds a bias to the data. The bias will be in the first (zero'th) column, and consist of ones.

        Arguments:
            x_data (np.array): Array of data to add the bias on

        Returns:
            x_data (np.array): The same data, but with a one-column in the zero'th column index.
        """
        return np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)


class OneVsRestLogReg:
    """
    One vs the rest logistic classifier.
    For a multiclass classification problem with c classes, this model trains c logistic regression models
    and predicts the one corresponding to the highest probability.
    The calculations are not vectorized with respect to the c logistic models, they are trained sequentially.

    Methods:
        train: Trains the linear regression, either analytically, with SGD or newtons method.
        predict_probabilities: Predict probabilities on data (sigmoid values).
        predict: Predict on data (0 or 1).
    """
    def __init__(self, n_classes=None, n_features=None, with_bias=True, verbose=False):
        """
        Initializes models according to the amount of features in the training-data.
        If not provided, will be set in the training.

        Arguments:
            n_classes (int): The amount of classes to be predicted.
            n_features (int): The amount of features in the training data (without the bias).
            with_bias (bool): Weather or not to add a bias (intercept) to the model.
            verbose (bool): If True, will print some intermediate output from gradient descent.
        """
        self.with_bias = with_bias
        self.verbose = verbose
        self.models = []
        self.n_features = n_features
        if n_classes is not None:
            self.n_classes
            for _ in range(n_classes):
                self.models.append(LogisticRegression(n_features, with_bias, verbose))

    def train(self, x_train, y_train, method="gradient_descent",
              eta=0.01, epochs=20, batch_size=1, tol=None, tol_rounds=5):
        """
        Trains model.
        This can be done with stochastic gradient descent (SGD) or with newtons method.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            method (str): Determines the method that will be used to find the betas.
                Must be in ["gradient_descent", "newton"].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        check_arrays(x_train, y_train, dims=[0])

        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_train = np.expand_dims(x_train, 1)

        if len(y_train.shape) == 1:  # 1d array, one-hot-encode
            y_train = integer_one_hot_encode(y_train)

        if self.models == []:  # Initialize models if it has not yet been done.
            self.n_classes = y_train.shape[1]
            for _ in range(self.n_classes):
                self.models.append(LogisticRegression(self.n_features, self.with_bias, self.verbose))

        for c in range(self.n_classes):
            self.models[c].train(x_train, y_train[:, c], method, None, eta, epochs, batch_size, tol, tol_rounds)

    def predict_probabilities(self, x_data, have_bias=False):
        """
        Predicts on data, for each of the five models..
        Prediction for a datapoint is sigmoid(beta * x).

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n x c) array of predictions made. Point [n, c] corresponds to the
                c'th models probability for the n'th input.
        """
        probs = np.zeros((x_data.shape[0], self.n_classes))  # Initalize probs
        for c in range(self.n_classes):
            probs[:, c] = self.models[c].predict_probabilities(x_data, have_bias)
        return probs

    def predict(self, x_data, have_bias=False):
        """
        Predicts on class on data.

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        probs = self.predict_probabilities(x_data, have_bias)
        preds = probs.argmax(axis=1)
        return preds


class MultinomialLogReg:
    """
    Class for implementing multinomial logistic regression.
    Trained with stochastic gradient descent (SGD). Vectorized, with explaination of dimension sizes.

    We assume the classes are either from 0, 1, 2, ..., c - 1, or one-hot-encoded.
    Bias vector is added into the x-matrix and included in the weight matrix if with_bias is True.

    Overview of sizes of dimensions:
    m: features (including bias)
    c: classes
    n: inputs

    x_train [n x m]
    y_train [n x c]
    weights [c x m]
    z = W @ X.T [n x c]  # Weighted sum
    s = softmax(z)  # Output probabilities
    preds = s.argmax(axis=1)  # Discrete predictions
    dW = ((Y - S).T [c x n] @ X [n x m] ) [c x m]  # Weight update

    Methods:
        train: Trains the linear regression with SGD.
        predict_probabilities: Predict probabilities on data (softmax values).
        predict: Predict on data (discrete classes).
    """
    def __init__(self, n_classes=None, n_features=None, with_bias=True, verbose=False):
        """
        Initializes betas parameters according to the amount of features in the training-data.
        If not provided, will be set in the training.

        Arguments:
            n_features (int): The amount of features in the training data (without the bias).
            with_bias (bool): Weather or not to add a bias (intercept) to the model.
            verbose (bool): If True, will print some intermediate output from gradient descent.
        """
        self.weights = None
        self.with_bias = with_bias
        self.verbose = verbose
        if n_features is not None and n_classes is not None:
            self.n_classes = n_classes
            self.n_features = n_features
            self.weights = np.zeros((n_classes, self.n_features + int(with_bias)))  # Initialize weights

    def train(self, x_train, y_train, eval_set=None, eta=0.01, epochs=20, batch_size=1, tol=None, tol_rounds=5):
        """
        Trains model.
        This can only be done with stochastic gradient descent (SGD)

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        check_arrays(x_train, y_train, dims=[0])  # Check that the amount of (n) points are equal.

        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_train = np.expand_dims(x_train, 1)

        if len(y_train.shape) == 1:  # 1d array, one-hot-encode
            y_train = integer_one_hot_encode(y_train)

        if eval_set is not None:  # Check and one-hot-encode eval-set.
            x_val, y_val = eval_set
            check_arrays(x_val, y_val, dims=[0])
            if len(x_val.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
                x_val = np.expand_dims(x_val, 1)
            if len(y_val.shape) == 1:  # 1d array, one-hot-encode
                y_val = integer_one_hot_encode(y_val)
            eval_set = (x_val, y_val)

        if self.weights is None:  # Initialize beta-parameters
            self.n_classes = y_train.shape[1]
            self.n_features = x_train.shape[1]
            self.weights = np.zeros((self.n_classes, self.n_features + int(self.with_bias)))  # Initialize weights

        if self.with_bias:  # Add bias
            x_train = self._add_bias(x_train)
        self._train_gradient_descent(x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds)

    def predict_probabilities(self, x_data, have_bias=False):
        """
        Predicts on data.
        Prediction for a datapoint is softmax(weights * x), and gives a vector of length c.

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        if len(x_data.shape) == 1:  # Turn (n)-array into (n x 1)-matrix
            x_data = np.expand_dims(x_data, 1)

        if not have_bias and self.with_bias:  # Add bias
            x_data = self._add_bias(x_data)
        # Dimensions: ( W [c x m] X.T [m x n] ).T [n x c]
        z = (self.weights @ x_data.T).T  # Compute the weighted sum for each datapoint.
        return softmax(z)

    def predict(self, x_data, have_bias=False):
        """
        Predicts on data, 0, 1, 2, ... for each respective class.
        This means picking the argmax of the softmax-probabilities.

        Arguments:
            x_data (np.array): (n x m) array of n points with m features to be predicted on.
            have_bias (bool): If False, will add a bias to the data.

        Returns:
            preds (np.array): (n) array of predictions made.
        """
        probs = self.predict_probabilities(x_data, have_bias)
        preds = probs.argmax(axis=1)
        return preds

    def _train_gradient_descent(self, x_train, y_train, eval_set, eta, epochs, batch_size, tol, tol_rounds):
        """
        Trains data with vanilla SGD.

        Arguments:
            x_data (np.array): (n x m) shaped array, representing n inputs with m features.
            y_data (np.array): (n) shaped array, representing each of the true targets.
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                eval_set = (x_val, y_val) of shapes [k x m] and [k].
            eta (float): The learning rate for gradient descent.
            epochs (int): The amount of time we loop through the training data.
            batch_size (int): How many input should be passed before the weight gets updated.
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
                When tol is not None, "epochs" can be view as "max-epochs".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
        """
        b = batch_size
        n = len(x_train)
        shuffle_indices = np.random.choice(n, n, replace=False)  # Shuffle training data
        x_train = x_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        self.train_losses = np.zeros(epochs)  # Store training stats
        self.train_accuracies = np.zeros(epochs)
        self.val_losses = np.zeros(epochs)  # Will only be filled in if eval_set is not None.
        self.val_accuracies = np.zeros(epochs)

        for n_epoch in range(epochs):
            for i in range(int(n / b)):  # Loop over all the minibatches for SGD
                upper_index = np.min((n, (i + 1) * b))  # Round of in case batch-size does not match up with n.
                n_data = upper_index - i * b  # Should be b unless last iteration
                batch = x_train[i * b: upper_index]  # Index batch
                targets = y_train[i * b: upper_index]  # Index labels
                targets_class = targets.argmax(axis=1)  # Non-one-hot-encoded
                pred_probs = self.predict_probabilities(batch, have_bias=True)  # Do the forward pass
                preds = self.predict(batch, have_bias=True)  # Do the forward pass
                # Dims: [eta / n_data (1) (targets - preds).T [c x n] @ batch (n x m)] (c x m)
                self.weights += (eta / n_data) * (targets - pred_probs).T @ batch  # Update weights
                self.train_losses[n_epoch] += find_multiclass_cross_entropy(pred_probs, targets) * n_data / n
                self.train_accuracies[n_epoch] += find_accuracy(preds, targets_class) * n_data / n

            self._update_val_loss(eval_set, n_epoch)  # Update eval set
            stop_training = self._check_tolerance(tol, tol_rounds, n_epoch)  # Check for criteria
            if stop_training:  # We have reached our desired tolerance, quit training
                return

        self.epochs_ran = n_epoch + 1  # Save the amount of epochs ran, for plotting

    def _update_val_loss(self, eval_set, n_epoch):
        """
        Updates eval-loss on eval-set, if it is not None.

        Arguments:
            eval_set (list / tuple): If not None, will do evaluation on the eval set after each epoch.
                    eval_set = (x_val, y_val) of shapes [k x m] and [k].
            n_epoch (int): The epoch number we are currently on.
        """
        if eval_set is not None:  # Update eval losses
            x_val, y_val = eval_set
            pred_probs = self.predict_probabilities(x_val)
            preds = self.predict(x_val)
            targets_class = y_val.argmax(axis=1)  # Non-one-hot-encoded
            self.val_losses[n_epoch] = find_multiclass_cross_entropy(pred_probs, y_val)
            self.val_accuracies[n_epoch] = find_accuracy(preds, targets_class)

    def _check_tolerance(self, tol, tol_rounds, n_epoch):
        """
        Check if we have reached our desired tolerance, and if we have, quit the training.

        Arguments:
            tol (float): If not None, will stop training after "tol_rounds" amount of epochs if
                the training loss have not been improved by "tol".
            tol_rounds (int): The amount of rounds that needs to be run without an improvement bigger
                than "tol" for the training to be stopped.
            n_epoch (int): The epoch number we are currently on.

        Returns:
            stop_training (bool): A boolean representing wether or not we have reached our desired tolerance.
        """
        if tol is not None and n_epoch > tol_rounds:  # Check if we have not improved above the given tolerance
            best_prev = self.train_losses[:(n_epoch - tol_rounds)].min()  # Best loss until tol_rounds epcohs ago
            best_recent = self.train_losses[(n_epoch - tol_rounds):n_epoch].min()  # Best loss the last tol_rounds
            if (best_prev - best_recent) < tol:  # We have not improved enough
                if self.verbose:
                    print(f"Early stopping after epoch {n_epoch + 1} with tolerance {tol}.")
                self.epochs_ran = n_epoch + 1
                return True
        else:
            return False

    def _add_bias(self, x_data):
        """
        Adds a bias to the data. The bias will be in the first (zero'th) column, and consist of ones.

        Arguments:
            x_data (np.array): Array of data to add the bias on

        Returns:
            x_data (np.array): The same data, but with a one-column in the zero'th column index.
        """
        return np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)


if __name__ == "__main__":
    pass
