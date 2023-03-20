"""
Graveyard for old function and classes removed, changed or improved.
"""
import numpy as np
import matplotlib.pyplot as plt


# Non-vectorized distance, scaling, mse and accuracy:
class Distance:
    """
    Class for keeping distance functions (metrics).
    """
    @staticmethod
    def distanceL1(input1, input2):
        """
        Calculate the L1 distance of input1 and input2. The inputs must be of the
        same lengths. This is defined as the sum over the absolute distances of each dimension.
        This is also called "manhattan distance".

        Arguments:
            input1 (np.array): (m)-dimensional array, first point.
            input2 (np.array): (m)-dimensional array, second point.

        Out:
            distance (float): The L1 distance between the first and the second point.
        """
        distance = 0
        for i in range(input1.shape[0]):
            distance += np.absolute(input1[i] - input2[i])
        return distance

    @staticmethod
    def distanceL2(input1, input2):
        """
        Calculate the L2 distance of input1 and input2. The inputs must be of the
        same lengths. This is defined as the squareroot of the sum of the squared
        distances of each dimenion.
        This is also called Euclidian distance

        Arguments:
            input1 (np.array): (m)-dimensional array, first point.
            input2 (np.array): (m)-dimensional array, second point.

        Out:
            distance (float): The L2 distance between the first and the second point.
        """
        distance = 0
        for i in range(input1.shape[0]):
            distance += np.square(input1[i] - input2[i])
        return np.sqrt(distance)


class Scaling:
    """
    Class for keeping scaling functions.
    """
    def min_max_scaler(data, find_params=True, mins=None, maxes=None, return_params=False):
        """
        Scales the data "data" according to a min-max-scaler.
        This scaler will for every input vector i and dimensions j, calculate
            x[i, j] = (x[i, j] - min_j)/(max_j - min_j)
        where min_j and max_j are the smallest and biggest value in the j'th dimension (of "data" or
        previously found). Values will be between 0 and 1 (if mins and maxes are found).
        If find_params is True, will find the mins and maxes according to "data", and retrun them.
        This is done with the training set. If find_params is False, will use previously calculated
        mins and maxes. This is done with validation and testing set.
        Arguments:
            data (np.array): (m x n)-dimensional array of n inputs with m features to be scaled.
            find_params (boolean): If True, will scale according to the min and max of each feature-dimension
                of the given data. If False, will scale according to some previously found parameters. Then
                "mins" and "maxes" should be provided.
            mins (np.array / None): (m)-dimensional data of previously found minimums of each feature.
            maxes (np.array / None): (m)-dimensional data of previously found maximums of each feature.
            return_params (boolean): If True, will return the mins and maxes.
        Returns:
            scaled_data (np.array): (m x n)-dimensional scaled array of n inputs with m features.
            if return_params:
                scaling_params (dict): {"mins": mins, "maxes": maxes} where
                    mins (np.array): (m)-dimensional data of caclulated minimums of each feature.
                    maxes (np.array): (m)-dimensional data of caclulated maximums of each feature.
        """
        if find_params:  # Find mins and maxes of each feature-dimension
            m = data.shape[1]  # Amount of features for each input
            mins = np.zeros(m)
            maxes = np.zeros(m)
            for j in range(data.shape[1]):  # loop over every feature dimension
                mins[j] = data[:, j].min()
                maxes[j] = data[:, j].max()

        scaled_data = np.zeros(data.shape)
        for i in range(data.shape[0]):  # Loop over every observation
            for j in range(data.shape[1]):  # Loop over every feature
                scaled_data[i, j] = (data[i, j] - mins[j]) / (maxes[j] - mins[j])

        if return_params:
            return scaled_data, {"mins": mins, "maxes": maxes}
        else:
            return scaled_data

    def normal_scaler(data, find_params=True, means=None, stds=None, return_params=False):
        """
        Scales the data "data" according to a normal / standard scaler.
        This scales every input in data by removing the mean (of the feature) and dividing by the
        std (of the feature), (x[i, j] = x[i, j] - mean_j) / std_j
        If find_params is True, will find the mins and maxes according to "data", and retrun them.
        This is done with the training set. If find_params is False, will use previously calculated
        mins and maxes. This is done with validation and testing set.
        Arguments:
            data (np.array): (m x n)-dimensional array of n inputs with m features to be scaled.
            find_params (boolean): If True, will scale according to the min and max of each feature-dimension
                of the given data. If False, will scale according to some previously found parameters. Then
                "mins" and "maxes" should be provided.
            means (np.array / None): (m)-dimensional data of previously found means of each feature.
            stds (np.array / None): (m)-dimensional data of previously found stds of each feature.
            return_params (boolean): If True, will return the mins and maxes.
        Returns:
            scaled_data (np.array): (m x n)-dimensional scaled array of n inputs with m features.
            if return_params:
                scaling_params (dict): {"means": means, "stds": stds} where
                    means (np.array): (m)-dimensional data of caclulated means of each feature.
                    stds (np.array): (m)-dimensional data of caclulated stds of each feature.
        """
        if find_params:  # Find means and stds based on the data provided
            m = data.shape[1]
            means = np.zeros(m)
            stds = np.zeros(m)
            for j in range(m):
                means[j] = data[:, j].mean()
                stds[j] = data[:, j].std()

        scaled_data = np.zeros(data.shape)
        for i in range(data.shape[0]):  # Look over inputs
            for j in range(data.shape[1]):  # Loop over features
                scaled_data[i, j] = (data[i, j] - means[j]) / stds[j]

        if return_params:
            return scaled_data, {"means": means, "stds": stds}
        else:
            return scaled_data

    @staticmethod
    def no_scaling(data, find_params=False, return_params=False):
        """
        This is a function that does no scaling. It is implemented so that the code
        that uses scaling can be more generic when it choses to not do scaling.
        Arguments:
            data (np.array): (m x n)-dimensional array of n inputs with m features to be scaled.
            find_params (boolean): Does not do anything, but is here to keep things using scaling generic.
            return_params (boolean): If True, will return an empty dictionary, for generic things using scaling.
        """
        if return_params:
            return data, {}
        else:
            return data


def find_accuracy(predictions, labels):
    """
    Calculates the accuracy of some predictions given its label.
    This is defined as (number of correct guesses) / (number of guesses).
    This works only for classification, because a predictions is only correct
    if they are compared equal.
    Arguments:
        predictions (np.array): (n)-dimensional array of predictions.
        labels (np.array): (n)-dimensional array of true labels
            for the same inputs as the predictions
    Returns:
        accuracy (float): The accuracy for the predictions
    """
    if predictions.shape != labels.shape:  # Check for compatible arrays
        message = f"The prediction's dimensions must be equal to the true labels dimensions."
        message += f"It was: Predictions: {predictions.shape} and labels: {labels.shape}. "
        raise ValueError(message)

    accuracy = 0
    for i in range(predictions.shape[0]):
        if np.array_equal(predictions[i], labels[i]):  # Count the correctly predicted classes
            accuracy += 1

    accuracy = accuracy / predictions.shape[0]  # Divide by number of observations
    return accuracy


def find_mse(predictions, labels):
    """
    Calculates Mean Squared Error (MSE).
    TODO: Fully vectorize this
    """
    if predictions.shape != labels.shape:  # Check for compatible arrays
        message = f"The prediction's dimensions must be equal to the true labels dimensions."
        message += f"It was: Predictions: {predictions.shape} and labels: {labels.shape}. "
        raise ValueError(message)

    mse = 0
    for i in range(predictions.shape[0]):
        mse += np.square(predictions[i] - labels[i])  # Calculate the square distance

    mse = mse / predictions.shape[0]  # Divide by number of observations
    return mse
