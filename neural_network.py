"""
File for implementing neural netowrks.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import (softmax, identity_function, find_mse, find_accuracy,
                    find_binary_cross_entropy, find_multiclass_cross_entropy, check_arrays, integer_one_hot_encode,
                    plot_decision_regions)
from activation_functions import Sigmoid, Tanh
from exceptions import NotTrainedError


"""
TODO:
Add gradient-checking
Save + Read weights
Add visualization for weight layers
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
    "*" Marks elementwise product. "@" marks matrix multiplication. "o" marks omuter product.

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

    def __init__(self, layer_sizes, model_type="multilabel", activation_functions=None, loss_function=None,
                 weight_initialization="plain", fan_mode="fan_in", threshold=0.5, dropouts=None):
        """
        Type-checks, initializes weights and biases and other stuff.
        Works for:
            Regression: Identity output function and mean square error loss function
            Multiclass: Softmax output function and multiclass cross entropy loss function
            Multilabel: Sigmoid output function and binary cross entropy loss function (on every node).
                Also works on mutliclass (mutlilabel with one class).

        Arguments:
            layer_sizes (list): Contains amount of nodes in each layer. Input size
                is the first element in this list, and output size is the last element.
            model_type (str): Which type of model. Must be in ["regression", "multilabel", "binary", "multiclass"].
                "regression" gives an identity output (last layer activation) function and
                MSE loss. "multilabel" gives sigmoid ouput function and binary cross entropy loss
                on every output node (BCEWithLogits). "binary" is the same as multilabel, but there is
                one output node. This means y-data does not get one-hot-econded. "multiclass" gives
                softmax output function and multiclass cross entropy loss.
            activation_function (list of callable): List of activation functions. Should not include
                the last activation function (output function), this is determined by "model_type".
            loss_function (callable): Loss function. This will only be used to calculate loss during
                training, since loss function for training will be assumed from "model_type".
            weight_initialization (str): How the weights should be initialized. Se _initialize_weights()
                for more information.
            fan_mode (str): Decides wether to use "fan_in" or "fan_out" in kaiming_he initialization.
                Must be in ["fan_in", "fan_out"].
            threshold (float): Threshold for predicting when using model_type as "mutlilabel".
            dropout (list of floats): If not None, will use dropout in the layers. Each layer l will have a keep
                probability of "outputs" l´th position, [p_1, ..., p_N], where p_i is on (0, 1].
                Note: Dropout significantly slows down training. The bottleneck is drawing bernulli data with
                np.random.binomial. I could not find any more efficiently implemented functions.
        """
        # Check types and stuff
        if not hasattr(layer_sizes, "__len__"):
            message = f"Argument \"layer_sizes\" must be list or array. Was {layer_sizes}, type {type(layer_sizes)}"
            raise ValueError(message)
        if activation_functions is not None and not hasattr(activation_functions, "__len__"):
            message = f"Argument \"activation_functions\" must be list or array. "
            message += f"Was {activation_functions}, type {type(activation_functions)}"
            raise ValueError(message)
        if model_type not in ["regression", "multilabel", "binary", "multiclass"]:
            message = f"Argument \"model_type\" must be in [\"regression\", \"multilabel\", \"binary\", \"multiclass\"]"
            message += f". Was {model_type}. "
            raise ValueError(message)
        if activation_functions is not None and (len(layer_sizes) - len(activation_functions)) != 2:
            message = f"Argument \"activation_functions\" must be of length 2 less than \"layer_sizes\". "
            message += f"Final layer activation function (output function) should not be included. "
            message += f"Was {len(activation_functions)} and {len(layer_sizes)}. "
            raise ValueError(message)
        if (model_type == "regression" or model_type == "binary") and layer_sizes[-1] != 1:
            message = "Nodes in output layer must be exactly 1 for regression or binary classification networks. "
            message += f"Was {self.layer_sizes[-1]}. "
            raise ValueError(message)
        if dropouts is not None:
            if not hasattr(dropouts, "__len__"):
                message = f"Argument \"dropouts\" must be list or array. "
                raise ValueError(message)
            if len(dropouts) != len(layer_sizes) - 1:
                message = f"Length of argument \"dropouts\" must be of length len(layer_sizes) - 1. "
                message += f"Length of \"dropouts\" was {len(dropouts)}, but length of \"layers_sizes\" "
                message += "was {len(layer_sizes)}. "
                raise ValueError(message)
            for p in dropouts:
                if not ((p <= 1) and (p > 0)):
                    message = f"Every argument of \"dropouts\" must be float in (0, 1]. Was {p}. "
                    raise ValueError(message)

        valid_weight_initializations = ["standard_normal", "plain", "glorot_uniform", "glorot_normal", "glorot",
                                        "xavier", "kaiming_he_uniform", "kaiming_he_normal", "kaiming_he", "he"]
        if weight_initialization not in valid_weight_initializations:
            message = f"Argument \"weight_initialization\" must be in {valid_weight_initializations}. "
            message += f"Was {weight_initialization}. "
            raise ValueError(message)
        if fan_mode not in ["fan_in", "fan_out"]:
            message = f"Argument \"fan_mode\" must be in [\"fan_in\", \"fan_out\"]. Was {fan_mode}. "
            raise ValueError(message)

        # Save class variables
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.threshold = threshold
        self.activation_functions = activation_functions
        if activation_functions is None:  # Set activation functions if they are not provided
            self.activation_functions = [Sigmoid()] * (self.n_layers - 2)
        self.dropouts = dropouts
        if dropouts is None:  # Set default value for dropouts, dropout = 1 means no dropout.
            self.dropouts = np.ones(self.n_layers - 1)

        self._initialize_weights(layer_sizes, weight_initialization, fan_mode)

        self.model_type = model_type
        if model_type == "regression":
            self.activation_functions.append(identity_function)  # Add output function (regression).
        elif model_type == "multiclass":
            self.activation_functions.append(softmax)  # Add output function (multiclass).
        elif model_type == "multilabel" or model_type == "binary":
            self.activation_functions.append(Sigmoid())  # Add output function (mutlilabel).

        self.loss_function = loss_function
        if loss_function is None:  # Set default loss function
            if self.model_type == "regression":
                self.loss_function = find_mse
            elif self.model_type == "multiclass":
                self.loss_function = find_multiclass_cross_entropy
            elif self.model_type == "multilabel" or model_type == "binary":
                self.loss_function = find_binary_cross_entropy
        self.epochs_ran = None  # Will be marked with number of epochs ran after training

    def train(self, x_train, y_train, eta=0.1, epochs=10, minibatch_size=64, evaluate=False, eval_set=None,
              lam=0, optimizer="SGD", correct_bias=False, beta1=0.9, beta2=0.999, epsilon=10e-4, verbose=False):
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
            lam (float): Lambda hyperparameter for managing L2 penalty for weight-updates.
            optimizer (str): The method use to update the weights.
                Must be in ["SGD", "momentum", "RMSprop", "Adam"]
            correct_bias (bool): If True, will do bias correctiong. This means scaling the
                moment so that the first updates are not too small (since the moment starts at 0).
            beta1 (float): The first moment parameter, specifying how much the momentum for the
                gradient is updated in "Momentum" and "Adam".
            beta2 (float): The second moment parameter, indicating how much the running average
                for RMSprop is updated in "RMSProp" and "Adam".
            epsilon (float): The stabilizing parameter in the denominator for "RMSProp" and "Adam".
            verbose (bool): If True, will print output.
        """
        # Initialize parameters
        self.lam = lam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum_b = []
        self.momentum_w = []
        self.second_moment_b = []
        self.second_moment_w = []
        for i in range(1, self.n_layers):
            self.momentum_b.append(np.zeros(self.layer_sizes[i]))
            self.momentum_w.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i - 1])))
            self.second_moment_b.append(np.zeros(self.layer_sizes[i]))
            self.second_moment_w.append(np.zeros((self.layer_sizes[i], self.layer_sizes[i - 1])))
        self._evaluate = evaluate
        self._preprocess(x_train, y_train, eval_set)

        if self._evaluate:  # Initialize losses and accuracies
            self.train_losses = np.zeros(epochs)
            if self.model_type != "regression":
                self.train_accuracies = np.zeros(epochs)
            if eval_set is not None:
                self.val_losses = np.zeros(epochs)
                if self.model_type != "regression":
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
                preds = self._forward(batch, training=True)  # Get logits (ouput-nodes) values, and save other values
                deltas = self._backprop(preds, targets)  # Get dela-error terms from backprop
                self._update_params(deltas, eta, n_data, optimizer, correct_bias, n_epoch + 1)  # Update parameters

            if self._evaluate:  # Calculate loss and optinally accuracies
                self._perform_evaluation(n_epoch, verbose)

        self.epochs_ran = n_epoch + 1  # Save epochs

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
        if self.model_type == "regression":
            return forwards
        elif self.model_type == "multiclass":  # Classification, return highes certainty prediction
            return forwards.argmax(axis=1)
        elif self.model_type == "multilabel" or self.model_type == "binary":  # Discretize
            return np.array(forwards > self.threshold, dtype=int)

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
        if self.model_type == "regression":  # We only plot loss
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
        if self.model_type != "regression":  # We also plot accuracy
            ax = fig.add_subplot(1, 2, 2)
            ax.plot(x_values, self.train_accuracies, label="Train-accuracies")
            if self._val:
                ax.plot(x_values, self.val_accuracies, label="Validation-accuracies")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.title.set_text("Accuracies over epochs")
            ax.legend()
            plt.tight_layout()

        if show:
            plt.show()

    def _initialize_weights(self, layer_sizes, method, fan_mode="fan_in"):
        """
        Initializes weights. "method" must be in:
        ["standard_normal", "plain", "glorot_uniform", "glorot_normal", "kaming_he_uniform", "kaiming_he_normal"]
        See https://pytorch.org/docs/stable/nn.init.html for more information, including reference to papers.

        Methods:
            standard_normal: Sets all weights to be drawn from standard normal distribution. Not recommended, since
                it does not look at amount of nodes in and out.
            plain: Draw all weights from N(0, 1/sqrt(n_in)).
            glorot_uniform: Draw all weights from U(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
            glorot_normal: Recommended for sigmoid or tanh. Draw all weights from N(0, sqrt(2 / (n_in + n_out)))
            kaiming_he_uniform: Draw all weights from U(-sqrt(6 / fan_mode), sqrt(6 / fan_mode))
            kaiming_he_normal: Recommended for ReLU and LeakyReLU. Draw all weights from N(0, sqrt(2 / fan_mode))

        Arguments:
            layers_sizes (list): List of int of nodes in each layer.
            method (str): Method to use, see above.
            fan_mode (str): Wether to use n_nodes_in or n_nodes_out for kaiming-he.
                Must be in ["fan_in", "fan_out"]
        """
        self.biases = [np.random.randn(1, layer_sizes[i]) for i in range(1, len(layer_sizes))]
        self.weights = []
        if method == "standard_normal":
            self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) for i in range(1, len(layer_sizes))]

        elif method == "plain":
            for i in range(len(self.layer_sizes) - 1):
                std = 1 / np.sqrt(layer_sizes[i])
                self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * std)

        elif method == "glorot_uniform":
            for i in range(len(self.layer_sizes) - 1):
                a = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                self.weights.append(np.random.uniform(-a, a, (layer_sizes[i + 1], layer_sizes[i])))

        elif method == "glorot_normal" or method == "glorot" or method == "xavier":
            for i in range(len(self.layer_sizes) - 1):
                std = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
                self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i]))

        elif method == "kaiming_he_uniform":
            for i in range(len(self.layer_sizes) - 1):
                if fan_mode == "fan_in":
                    a = np.sqrt(6 / layer_sizes[i])
                elif fan_mode == "fan_out":
                    a = np.sqrt(6 / layer_sizes[i + 1])
                self.weights.append(np.random.uniform(-a, a, (layer_sizes[i + 1], layer_sizes[i])))

        elif method == "kaiming_he_normal" or method == "kaiming_he" or method == "he":
            for i in range(len(self.layer_sizes) - 1):
                if fan_mode == "fan_in":
                    std = np.sqrt(2 / layer_sizes[i])
                elif fan_mode == "fan_out":
                    std = np.sqrt(2 / layer_sizes[i + 1])
                self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i]))

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
        check_arrays(x_train, y_train, dims=[0])
        if len(x_train.shape) == 1:  # Turn (n)-array into (n x 1)-matrix (only one feature)
            x_train = np.expand_dims(x_train, 1)

        if len(y_train.shape) == 1:  # 1d array, either expand dims or one-hot-encode
            if self.model_type == "regression" or self.model_type == "binary":  # Add axis
                y_train = np.expand_dims(y_train, 1)
            else:  # Classification or multilabel, one hot encode
                if y_train.max() >= self.layer_sizes[-1]:
                    message = "Classes provided in y_train exceeds amount of output-nodes. "
                    message += "If y_train classes does not contain class value from 0, ..., c-1, it must be "
                    message += "one-hot encoded before sent to train(). "
                    message += f"y_trains max element was {y_train.max()} and output nodes were {self.layer_sizes[-1]}."
                    raise ValueError(message)
                y_train = integer_one_hot_encode(y_train, self.layer_sizes[-1] - 1)
        # Save variables
        self.x_train = x_train
        self.y_train = y_train

        # Validation sets:
        self._val = False
        if eval_set is not None:
            self._val = True  # We have provided validation set (for plotting later on)
            self._evaluate = True  # Always evaluate if eval_set is provided
            x_val, y_val = eval_set
            check_arrays(x_val, y_val, dims=[0])
            if len(x_val.shape) == 1:  # Turn (n)-array into (n x 1)-matrix (only one feature)
                x_val = np.expand_dims(x_val, 1)
            if len(y_val.shape) == 1:  # 1d array, either expand dims or one hot encode
                if self.model_type == "regression" or self.model_type == "binary":  # Add axis
                    y_val = np.expand_dims(y_val, 1)
                else:  # Classification or multilabel, one-hot-encode
                    if y_val.max() >= self.layer_sizes[-1]:
                        message = "Classes provided in y_val exceeds amount of output-nodes. "
                        message += "If y_val classes does not contain class values from 0, ..., c-1, it must be "
                        message += "one-hot encoded before sent to train(). "
                        message += f"y_vals max element was {y_val.max()} and output nodes were {self.layer_sizes[-1]}."
                        raise ValueError(message)
                    y_val = integer_one_hot_encode(y_val, self.layer_sizes[-1] - 1)
            self.x_val = x_val  # Set validation class variables
            self.y_val = y_val

        # Shuffle training-set
        n = x_train.shape[0]
        shuffle_indices = np.random.choice(n, n, replace=False)  # Shuffle training data
        self.x_train = self.x_train[shuffle_indices]
        self.y_train = self.y_train[shuffle_indices]
        self._check_arrays()

    def _check_arrays(self):
        """
        Check that arrays are of correct sizes.
        Check if feature in x-data match with input nodes.
        """
        if self.x_train.shape[1] != self.layer_sizes[0]:
            message = "Features in training data (x_train) must mach nodes in first layer (layer_sizes). "
            message += f"Was {self.x_train.shape[1]} and {self.layer_sizes[0]}. "
            raise ValueError(message)
        if self._evaluate:
            if self.x_val.shape[1] != self.layer_sizes[0]:
                message = "Features in validation data (x_val) must mach nodes in first layer (layer_sizes). "
                message += f"Was {self.x_val.shape[1]} and {self.layer_sizes[0]}. "
                raise ValueError(message)

    def _forward(self, x_data, training=False):
        """
        Feeds the data forward. Do not discretize the output (give logits values).
        For each layer, calculates the activations, and feeds forward.

        Arguments:
            x_data (np.array): [n x m] data to forward.
            training (boolean): Mark True if forwarded during training, else False.
                If True, will enable (inversed) dropout.

        Returns:
            activations (np.array): [n x c] array over logits outputs (activations of last layer).
        """
        self.weighted_sums = []  # Save weighted sums for backpropagation
        self.activations = []  # Save all the activations for backpropagation
        self.dropout_coeffs = []  # Save the dropout coeffs for backprogatation step
        if training and self.dropouts[0] != 1:  # Perform dropout on inputs
            p = self.dropouts[0]
            # bernullis = np.random.binomial(1, p, (x_data.shape[0], self.layer_sizes[0]))
            # For some reason this is faster than binomial
            bernullis = np.random.rand(x_data.shape[0], x_data.shape[1]) < p
            dropout_coeff = bernullis / p  # Do the scaling in trainig, inverse dropout (and converts bool to int)
            x_data = dropout_coeff * x_data  # Perform the dropout elementwise multiplication
            self.dropout_coeffs.append(dropout_coeff)
        else:
            self.dropout_coeffs.append(np.ones_like(x_data))  # Save ones incase no dropout in first layer
        self.activations.append(x_data)
        for i in range(self.n_layers - 1):
            # z(l) [n x n(l)] = a(l-1) [n x n(l-1)] @ w(l) [n(l) x n(l-1)].T + b(l) [1, n(l)]
            weighted_sum = self.activations[i] @ self.weights[i].T + self.biases[i]
            activation = self.activation_functions[i](weighted_sum)  # a(l) = s(z(l))

            if training and self.dropouts[i] != 1 and i < (self.n_layers - 2):  # Perform dropout (not on last layer)
                p = self.dropouts[i]
                # bernullis = np.random.binomial(1, p, (x_data.shape[0], self.layer_sizes[i + 1]))
                # For some reason this is faster than binomial
                bernullis = np.random.rand(activation.shape[0], activation.shape[1]) < p
                dropout_coeff = bernullis / p  # Do the scaling in trainig, inverse dropout
                activation = dropout_coeff * activation  # Perform the dropout elementwise multiplication
                self.dropout_coeffs.append(dropout_coeff)
            elif i < (self.n_layers - 2):
                self.dropout_coeffs.append(np.ones_like(activation))  # Save ones if no dropout here, but other places
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
        delta_L = self._delta_L(preds, targets)
        deltas.append(delta_L)
        for i in range(1, self.n_layers - 1):
            index = self.n_layers - i - 1
            prev_delta = deltas[0]  # First element is the previous layers delta
            # del(l) [n x n(l)] = (del(l+1) [n x n(l+1)] @ w(l+1) [n(l+1) x n(l)]) [n x n(l)] * s'(z(l)) [n x n(l)]
            delta = prev_delta @ self.weights[index]
            delta = delta * self.activation_functions[-(i + 1)].diff(self.weighted_sums[index - 1])
            if self.dropouts[-i] != 1:  # Multiply with the dropout coeffs:
                delta = delta * self.dropout_coeffs[-i]
            deltas.insert(0, delta)
        return deltas

    def _update_params(self, deltas, eta, n_data, optimizer="SGD", correct_bias=False, t=999):
        """
        Update every parameter, given the delta values. Calls the function corresponding to
        the optimizer.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
            optimizer (str): The method use to update the weights.
                Must be in ["SGD", "momentum", "RMSprop", "Adam"]
            correct_bias (bool): If True, will do bias correctiong. This means scaling the
                moment so that the first updates are not too small (since the moment starts at 0).
            t (int): The timestep the update happens at. This is used for bias-correction.
        """
        if optimizer.lower() in ["sgd", "vanilla", "vanilla_sgd"]:
            self._sgd(deltas, eta, n_data)
        elif optimizer.lower() in ["momentum", "moment"]:
            self._momentum(deltas, eta, n_data, correct_bias, t)
        elif optimizer.lower() in ["rms", "rmsprop", "rms-prop", "rms_prop", "rms-propagation", "rms_propagation"]:
            self._rms_prop(deltas, eta, n_data, correct_bias, t)
        elif optimizer.lower() in ["adam", "adamw"]:
            self._adam(deltas, eta, n_data, t)
        else:
            message = f"Argument \"optimizer\" to NeuralNetowrk.train() must be in "
            message += f"[\"SGD\", \"momentum\", \"RMSprop\", \"adam\"], was {optimizer}. "
            raise ValueError(message)

    def _sgd(self, deltas, eta, n_data):
        """
        Uses vanilla Stochastic Gradient Descent (SGD) to update parameters.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            d_biases = deltas[i].sum(axis=0)  # Normal update term
            l2_term_biases = self.lam * self.biases[i]  # L2 Regularization term
            self.biases[i] -= (eta / n_data) * (l2_term_biases + d_biases)  # With L2 regularization term

            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            d_weights = deltas[i].T @ self.activations[i]  # Normal update term
            l2_term_weights = self.lam * self.weights[i]  # L2 regularization term
            self.weights[i] -= (eta / n_data) * (l2_term_weights + d_weights)  # With L2 regularization

    def _momentum(self, deltas, eta, n_data, correct_bias=False, t=999):
        """
        Uses conventional momentum (CM) with SGD to update parameters.
        NOTE: The bias-correction does not really work, except if the learning rate is way to low.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch.
            correct_bias (bool): If True, will do bias correctiong. This means scaling the
                moment so that the first updates are not too small (since the moment starts at 0).
            t (int): The timestep the update happens at. This is used for bias-correction.
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            d_biases = deltas[i].sum(axis=0)  # Normal update term
            self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * d_biases
            mt_b = self.momentum_b[i]
            if correct_bias:
                mt_b /= (1 - self.beta1 ** t)
            l2_term_biases = self.lam * self.biases[i]  # L2 Regularization term
            self.biases[i] -= (eta / n_data) * (l2_term_biases + mt_b)  # With L2 regularization term

            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            d_weights = deltas[i].T @ self.activations[i]  # Normal update term
            self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * d_weights
            mt_w = self.momentum_w[i]
            if correct_bias:
                mt_w /= (1 - self.beta1 ** t)
            l2_term_weights = self.lam * self.weights[i]  # L2 regularization term
            self.weights[i] -= (eta / n_data) * (l2_term_weights + mt_w)  # With L2 regularization

    def _rms_prop(self, deltas, eta, n_data, correct_bias=True, t=999):
        """
        Uses RMS prop to updeight parameters.
        NOTE: The bias-correction does not really work, except if the learning rate is way to low.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
            correct_bias (bool): If True, will do bias correctiong. This means scaling the
                moment so that the first updates are not too small (since the moment starts at 0).
            t (int): The timestep the update happens at. This is used for bias-correction.
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            d_biases = deltas[i].sum(axis=0)  # Normal update term
            self.second_moment_b[i] = self.beta2 * self.second_moment_b[i] + (1 - self.beta2) * np.square(d_biases)
            vt_b = self.second_moment_b[i]
            if correct_bias:
                vt_b /= (1 - self.beta2 ** t)  # Bias correct
            l2_term_biases = self.lam * self.biases[i]  # L2 Regularization term
            self.biases[i] -= (eta / n_data) * (l2_term_biases + d_biases) / (np.sqrt(vt_b) + self.epsilon)

            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            d_weights = deltas[i].T @ self.activations[i]  # Normal update term
            self.momentum_w[i] = self.beta2 * self.momentum_w[i] + (1 - self.beta2) * np.square(d_weights)
            vt_w = self.momentum_w[i]
            if correct_bias:
                vt_w /= (1 - self.beta2 ** t)  # Bias correct
            l2_term_weights = self.lam * self.weights[i]  # L2 regularization term
            self.weights[i] -= (eta / n_data) * (l2_term_weights + d_weights) / (np.sqrt(vt_w) + self.epsilon)

    def _adam(self, deltas, eta, n_data, t=999):
        """
        Uses Adaptive Moment estimation (ADAM) to updeight parameters.

        Arguments:
            deltas (list): The delta values returned from _backprop.
            eta (float): Learning rate of the optimizer.
            n_data (int): Amount of datapoints used in minibatch
            t (int): Timestep the update happens in. This is used for bias-correction.
        """
        for i in range(self.n_layers - 1):
            # dC/db(l) [n(l)] = del(l) [n x n(l)].sum(axis=0)
            d_biases = deltas[i].sum(axis=0)  # Normal update term
            self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * d_biases  # First moment
            mt_b = self.momentum_b[i] / (1 - self.beta1 ** t)  # First moment With bias correction
            self.second_moment_b[i] = self.beta2 * self.second_moment_b[i] + (1 - self.beta1) * np.square(d_biases)
            vt_b = self.second_moment_b[i] / (1 - self.beta2 ** t)  # Second moment With bias correction
            l2_term_biases = self.lam * self.biases[i]  # L2 Regularization term
            self.biases[i] -= (eta / n_data) * (l2_term_biases + mt_b) / (np.sqrt(vt_b) + self.epsilon)

            # dC/dw(l) [n(l) x n(l-1)] = del(l).T [n(l) x n] @ a(l-1) [n x n(l-1)]
            d_weights = deltas[i].T @ self.activations[i]  # Normal update term
            self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * d_weights  # First moment
            mt_w = self.momentum_w[i] / (1 - self.beta1 ** t)  # First moment With bias correction
            self.second_moment_w[i] = self.beta2 * self.second_moment_w[i] + (1 - self.beta1) * np.square(d_weights)
            vt_w = self.second_moment_w[i] / (1 - self.beta2 ** t)  # Second moment With bias correction
            l2_term_weights = self.lam * self.weights[i]  # L2 Regularization term
            self.weights[i] -= (eta / n_data) * (l2_term_weights + mt_w) / (np.sqrt(vt_w) + self.epsilon)

    def _perform_evaluation(self, n_epoch, verbose):
        """
        Perform evaluation of loss and optinal accuracy, on both train set and optinal evaluation.

        Arguments:
            n_epoch (int): The epoch that we are on.
            verbose (bool): If True (and validation set is provided), will print stats.
        """
        train_logits = self._forward(self.x_train)
        self.train_losses[n_epoch] = self.loss_function(train_logits, self.y_train)
        if self.model_type != "regression":  # Also calculate accuracy
            train_preds = self.predict(self.x_train)
            if self.model_type == "multiclass":
                train_preds = integer_one_hot_encode(train_preds, self.layer_sizes[-1] - 1)
            self.train_accuracies[n_epoch] = find_accuracy(train_preds, self.y_train)

        if self._val is not None:  # Stats for validation set
            val_logits = self._forward(self.x_val)
            self.val_losses[n_epoch] = self.loss_function(val_logits, self.y_val)
            if self.model_type != "regression":  # Also calculate accuracy
                val_preds = self.predict(self.x_val)
                if self.model_type == "multiclass":
                    val_preds = integer_one_hot_encode(val_preds, self.layer_sizes[-1] - 1)
                self.val_accuracies[n_epoch] = find_accuracy(val_preds, self.y_val)
            if verbose:
                print(f"Train-loss: {self.train_losses[n_epoch]:.5f}, ", end="")
                print(f"Validation-loss: {self.val_losses[n_epoch]:.5f}")
                if self.model_type != "regression":
                    print(f"Train-accuracy: {self.train_accuracies[n_epoch]:.5f}, ", end="")
                    print(f"Validation-accuracy: {self.val_accuracies[n_epoch]:.5f}")

    def _delta_L(self, preds, targets):
        """
        Return the delta term of the last layer, del_L = dC/da(L) * s'(z(L)).
        For regression (identity last-layer activation function and MSE loss),
        multiclass classifications with softmax, and multiclass or multilabel
        classification with sigmoid output layer (where we use binary cross entropy
        loss on every of the output nodes).
        All these derivations "magically" calculates to the same expression, a(L) - y.

        Arguments:
            preds (np.array): [n x c] array of predicted outputs (logits).
            targets (np.array): [n x c] array of true labels (one-hot-encoded for classification)

        Returns:
            delta_L (np.array): [n x c] array of the delta term in the last layer.
        """
        return preds - targets


if __name__ == "__main__":
    pass
