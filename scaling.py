"""
File containing implementations of scalers.
Every scaler inherints from "BaseScaler".
"""
import numpy as np
from exceptions import NotFittedError


class BaseScaler:
    """
    Base methods for all of the scalers.

    The scaling functions and classes are influenced by sklearn,
    https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/preprocessing/_data.py#L644
    """
    def __init__(self):
        self.params = None  # Initialize parameters used for scaling.

    def fit_transform(self, x_data):
        """
        Both finds the scaling parameters with respect to x_data and scales it.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features that will be scaled, according to the
                parameters found by fitting the data.

        Returns:
            scaled_data (np.array): (n x m) array of n inputs with m features. Scaled version of x_data.
        """
        self.fit(x_data)
        return self.transform(x_data)

    def check_if_fitted(self):
        """
        Checks if the scaler is fitted. If not, will raise NotFittedError.
        """
        if self.params is None:
            raise NotFittedError("Scaler must be fitted before transforming or getting parameters")

    def get_params(self):
        """
        Return the parameters gotten from fitting.

        Returns:
            params (dict): The parameters found during fitting.
        """
        self.check_if_fitted()
        return self.params


class MinMaxScaler(BaseScaler):
    """
    Scales the data "data" according to a min-max-scaler.

    This scaler will for every input vector i and dimensions j, calculate
        x[i, j] = (x[i, j] - min_j)/(max_j - min_j)
    where min_j and max_j are the smallest and biggest value in the j'th dimension (of "data" or
    previously found). Values will be between 0 and 1 (if mins and maxes are found).
    """
    def fit(self, x_data):
        """
        Fits the transforming. This means finding the parameters that scales the data.
        This means finding the mins and maxes for each feature.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features. This data determines how the
                data to transform() is scaled.
        """
        mins = np.expand_dims(x_data.min(axis=0), 0)  # Expand dims to broadcast on x_dat alater
        maxes = np.expand_dims(x_data.max(axis=0), 0)
        self.params = {"mins": mins, "maxes": maxes}

    def transform(self, x_data):
        """
        Actually does the scaling. fit() must be called already.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features that will be scaled, according to parameters
                already found.

        Returns:
            scaled_data (np.array): (n x m) array of n inputs with m features. Scaled version of x_data.
        """
        self.check_if_fitted()
        scaled_data = (x_data - self.params["mins"]) / (self.params["maxes"] - self.params["mins"])
        return scaled_data


class StandardScaler(BaseScaler):
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
    def fit(self, x_data):
        """
        Fits the transforming. This means finding the parameters that scales the data.
        This means saving the means and standard deviation for each feature.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features. This data determines how the
                data to transform() is scaled.
        """
        means = np.expand_dims(x_data.mean(axis=0), 0)  # Expand dims so that we can broadcast later.
        stds = np.expand_dims(x_data.std(axis=0), 0)
        self.params = {"means": means, "stds": stds}

    def transform(self, x_data):
        """
        Actually does the scaling. fit() must be called already.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features that will be scaled, according to parameters
                already found.

        Returns:
            scaled_data (np.array): (n x m) array of n inputs with m features. Scaled version of x_data.
        """
        self.check_if_fitted()
        scaled_data = (x_data - self.params["means"]) / self.params["stds"]
        return scaled_data


class NoScaler(BaseScaler):
    """
    This is a function that does no scaling. It is implemented so that the code
    that uses scaling can be more generic when it choses to not do scaling.
    """
    def fit(self, x_data):
        """
        Simple makes the dict for the parameters, since no scaling will be done.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features.
        """
        self.params = {}

    def transform(self, x_data):
        """
        Does the scaling, which is nothing.

        Arguments:
            x_data (np.array): (n x m) array of n inputs with m features that will be scaled.

        Returns:
            scaled_data (np.array): (n x m) array of n inputs with m features. Scaled version of x_data.
        """
        self.check_if_fitted()
        return x_data
