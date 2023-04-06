"""
Contain classes of activation functions that can be used for neural nets.
"""
import numpy as np


class Sigmoid:
    def __call__(self, x_array):
        """
        Performs the sigmoid function on every element in x_array.

        Arguments:
            x_array (np.array): (n) array of float values.

        Returns:
            simoids (np.array): (n) array of the sigmoid function, every element is between 0 and 1.
        """
        return 1 / (1 + np.exp(-x_array))

    def diff(self, x_array):
        """
        Returns derivative of the sigmoid functions, s'(x) = s(x) * (1 - s(x))

        Arguments:
            x_array (np.array): (n) array of float values.

        Returns:
            diff (np.array): (n) array of the derivatives of the sigmoid function.
        """
        forward = self(x_array)
        return forward * (1 - forward)


class Tanh:
    def __call__(self, x_array):
        """
        Hyperbolic tangens function, tanh. Defined as:
        tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))

        Arguments:
            x_array (np.array): (n) array of float values.

        Returns:
            tanhs (np.array): (n) array of the tanh function, every element is between 0 and 1.
        """
        return (np.exp(x_array) - np.exp(-x_array)) / (np.exp(x_array) + np.exp(-x_array))

    def diff(self, x_array):
        """
        Returns derivative of the tanh function

        Arguments:
            x_array (np.array): (n) array of float values.

        Returns:
            diff (np.array): (n) array of the derivatives of the tanh function.
        """
        forward = self(x_array)
        return 1 - np.square(forward)


class ReLU:
    pass


class LeakyReLU:
    pass
