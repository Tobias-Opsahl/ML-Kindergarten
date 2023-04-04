"""
File for implementing neural netowrks.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import (sigmoid, softmax, identity_function, find_mse, find_accuracy, 
    find_binary_cross_entropy, find_multiclass_cross_entropy, check_arrays, integer_one_hot_encode,
    plot_decision_regions)
from exceptions import NotTrainedError


"""
TODO:
Deal with derivatives of activations and loss
Add softmax
Add weight-initializaton
Add gradient-checking
Save + Read weights
Add regularization
Add optimizers
"""
class NeuralNetwork:
    """
    Class implementing vanilla fully connected neural network.
    It can be trained with stochastic gradient descent (SGD) with backpropagation.
    Backpropagation is not implemented as automatic differentiation, but as a
    backward pass function.

    By convention after http://neuralnetworksanddeeplearning.com/chap2.html,
    weight matrices have next layer as rows, and previous layers as columns [n(l) x n(l-1)]

    Notation and sizes of arrays:
    []-marks sizes of dimensions. "d" marks derivative.
    "*" Marks elementwise product. "@" marks matrix multiplication. "o" marks outer product.

    n: Amount of (mini-batch) inputs
    m: Amount of features in the inputs
    c: Amount of classes to predict (nodes in output layer)
    n(l): Amount of nodes in layer l.
    L: Amount of total layers.
    C(y; p): Cost function (loss function)

    x: [n x m] (mini-batch) input array
    p: [n x c] predictions, output array
    y: [n x c] true targets

    w(l) [n(l) x n(l-1)] weights for layer l (note the order of rows and colums
    b(l) [n(l) x 1] bias for layer l.
    z(l) [n x n(l)] weighted sum in layer l, z(l) = w(l) @ a(l) + b(l) (b(l) is broadcasted)
    a(l) [n x n(l)] activations in layer l, activation function of z, a = s(z), a(l) = s(l)(z(l))
    s(l) [callable] activation function in layer l.

    delta term, denoted "del(l)". Partial derivative denoted with "d", dy/dx
    Definition of delta terms: del(l) = dC / dz(l)

    Backpropagation formulas (non-vectorized, over index j and k):
    del(L)_j = dC/dp_j * s'(z(L)_j)
    del(l)_j = s'(z(l)_j) sum_k (w(l+1)_[k, j] del(l+1)_k)
    dC/db(l)_j = del(l)_j
    dC/dw(l)_[k, j] = del(l)_k * a(l-1)_j
    
    Backpropagation formulas (vectorized over layers):
    del(L) [c] = dC/dp [c] * s'(z(L)) [c]
    del(l) [n(l)] = (w(l+1).T [n(l) x n(l+1)] @ del(l+1) [n(l+1)]) [n(l)] * s'(z(l)) [n(l)]
    dC/db(l) [n(l)] = del(l) [n(l)]
    dC/dw(l) [n(l) x n(l-1)]= del(l) [n(l)] o a(l-1) [n(l-1)]

    Backpropagation formulas (vectorized over layers and minibatch):
    del(L) [n x c] = dC/dp [n x c] * s'(z(L)) [n x c]
    del(l) [n x n(l)] = (w(l+1).T [n(l) x n(l+1)] del(l+1).T [n(l+1) x n)]).T [n x n(l)] * s'(z(l)) [n x n(l)]
    del(l) [n x n(l)] = (del(l+1) [n x n(l+1)] @ w(l+1) [n(l+1) x n(l)]) [n x n(l)] * s'(z(l)) [n x n(l)]
    Weight and bias updates are summed over the n inputs:
    dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
    dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
    """

    def __init__(self, layer_sizes, regression=False, activation_functions=None, loss_function=None):
        """
        Initializes weights and biases.

        Arguments:
            layer_sizes (list): Contains amount of nodes in each layer. Input size
                is the first element in this list, and output size is the last element.
            regression (bool): If True, will train for regression. Will not one-hot-encode y.
            activation_function (list of callable): List of activation functions.
            loss_function (callable): Loss function.
        """
        if not hasattr(layer_sizes, "__len__"):
            message = f"Argument \"layer_sizes\" must be list or array. Was {layer_sizes}, type {type(layer_sizes)}"
            raise ValueError(message)
        # Save class variables
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.regression = regression
        self.biases = [np.random.randn(1, layer_sizes[i]) for i in range(1, len(layer_sizes))]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) for i in range(1, len(layer_sizes))]
        if activation_functions is None:
            if regression:
                self.activation_functions = [sigmoid] * (self.n_layers - 2) + [identity_function]
            else:
                self.activation_functions = [sigmoid] * (self.n_layers - 1)
        if loss_function is None:  # Set default loss function
            if self.regression:
                self.loss_function = find_mse
            else:
                self.loss_function = find_multiclass_cross_entropy
        self.epochs_ran = None  # Will be marked with number of epochs ran after training

    def train(self, x_train, y_train, eta=0.1, epochs=10, minibatch_size=64, evaluate=False, eval_set=None, verbose=False):
        """
        Trains network.

        Arguments:
            x_train (np.array): [n x m] input data of n inputs and m features.
            y_train (np.array): [n] or [n x c] array of true targets, either one-hot-encoded or not.
                If not one-hot-encoded, should be of shape (n, ) and contain labels (0, ..., c-1).
            eta (float): Learning rate of the optimizer.
            epochs (int): The amount of epochs (iterations over all data) to train for.
            minibatch_size (int): Size of mini-batch for SGD.
            evaluate (bool): If True, will calculate training loss and accuracy after each epoch.
            eval_set (tuple): If not None, will calculate validation loss and accuracy after each epoch.
                eval_set = (x_val, y_val). Will overide "evaluate" to True
            verbose (bool): If True, will print output.
        """
        self._evaluate = evaluate
        self._preprocess(x_train, y_train, eval_set)

        if self._evaluate:  # Initialize losses and accuracies
            self.train_losses = np.zeros(epochs)
            if not self.regression:
                self.train_accuracies = np.zeros(epochs)
            if eval_set is not None:
                self.val_losses = np.zeros(epochs)
                if not self.regression:
                    self.val_accuracies = np.zeros(epochs)

        b = minibatch_size
        n = x_train.shape[0]

        for n_epoch in range(epochs):
            if verbose:
                print(f"Epoch number {n_epoch}")
            for i in range(int(n / b)):  # Loop over all the minibatches for SGD
                upper_index = np.min((n, (i + 1) * b))  # Round of in case batch-size does not match up with n.
                n_data = upper_index - i * b  # Should be b unless last iteration
                batch = self.x_train[i * b: upper_index]  # Index batch
                targets = self.y_train[i * b: upper_index]  # Index targets
                preds = self._forward(batch)  # Get logits (ouput-nodes) values
                deltas = self._backprop(preds, targets)  # Get dela-error terms from backprop
                self._update_params(deltas, eta, n_data)  # Update weights and biases

            if self._evaluate:  # Calculate loss and optinally accuracies
                self._perform_evaluation(n_epoch, verbose)

        self.epochs_ran = n_epoch + 1 # Save epochs

    def predict(self, x_data):
        """
        Predicts on data, outputs the classes predicted.

        Arguments:
            x_data (np.array): [n x m] data to predict on.

        Returns:
            preds (np.array): [n] array of predicted classes (not one-hot-encoded).
        """
        if len(x_data.shape) == 1:  # Turn (m)-array into (1 x m)-matrix
            x_data = np.expand_dims(x_data, 0)
        forwards = self._forward(x_data)
        if self.regression:
            return forwards
        else:  # Classification, return highes certainty prediction
            return forwards.argmax(axis=1)
        
    def plot_stats(self, show=True):
        """
        Plots training losses, and validation losses if included.
        If classification, also plots accuracies.

        Arguments:
            show (bool): If True, calls plt.show().
        """
        if self.epochs_ran is None:
            message = f"Training must be called before calling plot_stats."
            raise NotTrainedError(message)
        
        fig = plt.figure()
        x_values = np.arange(self.epochs_ran)
        if self.regression:
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_values, self.train_losses, label="Train-losses")
        if self._val:
            ax.plot(x_values, self.val_losses, label="Validation-losses")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.title.set_text("Loss over epochs")
        ax.legend()
        if not self.regression:
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(x_values, self.train_accuracies, label="Train-accuracies")
            if self._val:
                ax.plot(x_values, self.val_accuracies, label="Validation-accuracies")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.title.set_text("Accuracies over epochs")
            ax.legend()
        
        if show:
            plt.show()
    
    def _preprocess(self, x_train, y_train, eval_set):
        """
        Preprocess training and optinally eval-sets.
        This only consist of managing dimensions and one-hot-encoding.
        This code is mainly for supporting both one-hot-encoding, non-one-hot-encoding and
        regression y's in both training and validations set.

        Arguments:
            x_train (np.array): [n x m] input data of n inputs and m features.
            y_train (np.array): [n] or [n x c] array of true targets, either one-hot-encoded or not.
                If not one-hot-encoded, should be of shape (n, ) and contain labels (0, ..., c-1).
            eval_set (tuple): If not None, will calculate validation loss and accuracy after each epoch.
                eval_set = (x_val, y_val). Will overide "evaluate" to True
        """
        # Train sets:
        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix (only one feature)
            x_train = np.expand_dims(x_train, 1)

        if self.regression:
            if len(y_train.shape) == 1:  # 1d array
                y_train = np.expand_dims(y_train, 1)  # Add axis
            
            self.x_train = x_train
            self.y_train = y_train
            self.y_train_score = y_train  # Targets to calculate loss is the same as predictions for regression
        else:  # Classification
            if len(y_train.shape) == 1:  # 1d array, one-hot-encode
                y_train_hot = integer_one_hot_encode(y_train)
            else:  # Already one-hot-encoded
                y_train_hot = y_train
                y_train = y_train_hot.argmax(axis=1)  # Non-one-hot encoding
            # Save variables
            self.x_train = x_train
            self.y_train = y_train_hot  # One-hot
            self.y_train_score = y_train  # Non-one-hot
        
        # Validation sets:
        self._val = False
        if eval_set is not None:
            self._val = True  # We have provided validation set (for plotting later on)
            self._evaluate = True  # Always evaluate if eval_set is provided
            x_val, y_val = eval_set
            check_arrays(x_val, y_val, dims=[0])
            if len(x_val.shape) == 1:  # Turn (n)-array into (n x 1)-matrix (only one feature)
                x_val = np.expand_dims(x_val, 1)

            if self.regression:
                if len(y_val.shape) == 1:  # 1d array
                    y_val = np.expand_dims(y_val, 1)  # Add axis
                self.x_val = x_val  # Set validation class variables
                self.y_val = y_val
                self.y_val_score = y_val
            else:  # Classification
                if len(y_val.shape) == 1:  # 1d array, one-hot-encode
                    y_val_hot = integer_one_hot_encode(y_val)
                else:  # Already one-hot-encoded
                    y_val_hot = y_val
                    y_val = y_val_hot.argmax(axis=1)  # Non-one-hot encoding
                self.x_val = x_val  # Set validation class variables
                self.y_val = y_val_hot
                self.y_val_score = y_val

        # Shuffle training-set
        n = x_train.shape[0]
        shuffle_indices = np.random.choice(n, n, replace=False)  # Shuffle training data
        self.x_train = self.x_train[shuffle_indices]
        self.y_train = self.y_train[shuffle_indices]
        self.y_train_score = self.y_train_score[shuffle_indices]

    def _forward(self, x_data):
        """
        Feeds the data forward. Do not discretize the output (give logits values).
        For each layer, calculates the activations, and feeds forward.

        Arguments:
            x_data (np.array): [n x m] data to forward.

        Returns:
            activations (np.array): [n x c] array over logits outputs (activations of last layer).
        """
        self.weighted_sums = []  # Save weighted sums for backpropagation
        self.activations = []  # Save all the activations for backpropagation
        self.activations.append(x_data)
        for i in range(self.n_layers - 1):
            # z(l) [n x n(l)] = a(l-1) [n x n(l-1)] @ w(l) [n(l) x n(l-1)].T + b(l) [1, n(l)]
            weighted_sum = self.activations[i] @ self.weights[i].T + self.biases[i]
            activation = self.activation_functions[i](weighted_sum)  # a(l) = s(z(l))
            self.weighted_sums.append(weighted_sum)
            self.activations.append(activation)
        return self.activations[-1]

    def _backprop(self, preds, targets):
        """
        Perform backpropagation, returns deltas for each layer.

        Arguments:
            preds (np.array): [b x c] predictied logits values.
            targets (np.array): [b x c] true target-values.

        Returns:
            deltas (list): List of deltas for each layer (except input layer).
        """
        deltas = []  # Add deltas to list backwards (insert at 0 when adding)
        # del(L) [n x c] = dC/dp [n x c] * s'(z(L)) [n x c]
        delta_L = self._loss_diff(preds, targets) * self._last_layer_diff(self.weighted_sums[-1])
        deltas.append(delta_L)
        for i in range(1, self.n_layers - 1):
            index = self.n_layers - i - 1
            prev_delta = deltas[0]  # First element is the previous layers delta
            # del(l) [n x n(l)] = (del(l+1) [n x n(l+1)] @ w(l+1) [n(l+1) x n(l)]) [n x n(l)] * s'(z(l)) [n x n(l)]
            delta = prev_delta @ self.weights[index]
            delta = delta * self._sigmoid_diff(self.weighted_sums[index-1])
            deltas.insert(0, delta)
        return deltas

    def _update_params(self, deltas, eta, n_data):
        """
        Update every parameter, given the delta values.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            self.biases[i] -= (eta / n_data) * deltas[i].sum(axis=0)
            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            self.weights[i] -= (eta / n_data) * deltas[i].T @ self.activations[i]

    def _perform_evaluation(self, n_epoch, verbose):
        """
        Perform evaluation of loss and optinal accuracy, on both train set and optinal evaluation.

        Arguments:
            n_epoch (int): The epoch that we are on.
            verbose (bool): If True (and validation set is provided), will print stats.
        """
        train_logits = self._forward(self.x_train)
        self.train_losses[n_epoch] = self.loss_function(train_logits, self.y_train)
        if not self.regression:  # Also calculate accuracy
            train_preds = self.predict(self.x_train)
            self.train_accuracies[n_epoch] = find_accuracy(train_preds, self.y_train_score)

        if self._val is not None:  # Stats for validation set
            val_logits = self._forward(self.x_val)
            self.val_losses[n_epoch] = self.loss_function(val_logits, self.y_val)
            if not self.regression:  # Also calculate accuracy
                train_preds = self.predict(self.x_val)
                self.val_accuracies[n_epoch] = find_accuracy(train_preds, self.y_val_score)
            if verbose:
                print(f"Train-loss: {self.train_losses[n_epoch]:.4f}, ", end="")
                print(f"Validation-loss: {self.val_losses[n_epoch]:.4f}")
                if not self.regression:
                    print(f"Train-accuracy: {self.train_accuracies[n_epoch]}, ", end="")
                    print(f"Validation-accuracy: {self.val_accuracies[n_epoch]}")

    # Derivatives of various functions. Should be re-organized for next version
    def _last_layer_diff(self, activations):
        if self.regression:
            return np.ones((activations.shape[0], 1))
        else:
            return self._sigmoid_diff(activations)

    def _sigmoid_diff(self, activations):
        return sigmoid(activations) * (1 - sigmoid(activations))
    
    def _loss_diff(self, preds, targets):
        return preds - targets
