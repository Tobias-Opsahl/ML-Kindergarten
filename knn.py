"""
k nearest neighbours methods.
Predicting based on the k closest points in the training set.
KNNClassifier and KNNRegressor.
"""

import numpy as np
from common import Distance, Scaling


class KNN:
    """
    Superclass for KNNClassifier and KNNRegressor.
    This method contains common methods that is used for both of them
    """
    def __init__(self, k=3, dist_func="L2", scaling="min_max"):
        """
        Arguments:
            k (int): The amount of nearest neighbours we look at.
            dist_func (str / callable): The function we use for calculating distances.
                Can be "L1" or "L2" for the L1 and L2 distances. Can also be callable.
                Then it should be a distance function that takes two arrays as input
            scaling (str): The scaler to use. Must be in ["min_max", "normal", "no_scaling"].
        """
        # Set k
        if not isinstance(k, (int, float, np.integer, np.floating)):
            raise TypeError(f"Argument \"k\" must be int. Was {type(k)}.")
        k = int(k)
        if k < 1:
            raise ValueError(f"Argument \"k\" must be 1 or higher. Was {k}.")
        self.k = k

        # Set distance-function
        if callable(dist_func):
            self._dist_func = dist_func
        elif dist_func == "L1":
            self._dist_func = Distance.distanceL1
        elif dist_func == "L2":
            self._dist_func = Distance.distanceL2
        else:
            raise ValueError(f"Argument \"dist_func\" must be \"L1\", \"L2\" or callable. Was {dist_func}.")

        # Set scaler
        if scaling == "min_max":
            self._scaler = Scaling.min_max_scaler
        elif scaling == "normal" or scaling == "standard":
            self._scaler = Scaling.normal_scaler
        elif scaling is False or scaling is None or scaling == "no_scaling":  # Set identity scaling (no scaling)
            self._scaler = Scaling.no_scaling
        else:
            message = f"Argument \"scaling\" must be \"min_max\", \"normal\" or \"no_scaling\", was {scaling}"
            raise ValueError(message)

    def train(self, x_train, y_train):
        """
        Stores the training set and the true labels. Scales

        Arguments:
            x_train (np.array): (m x n)-dimensional array of n inputs with m features.
            y_train (np.array): (n)-dimensional array of true label classes.
        """
        self.x_train, self._scaling_params = self._scaler(x_train, return_params=True)
        self.y_train = y_train
        self.m = len(x_train[0])  # Amount of features in one imput
        self.n = len(x_train)  # Amount of training inputs

    def _choose_from_neighbours(self, neighbours):
        """
        This is a method for chosing the predicted output from the k-nearest neighbours, "neighbours".
        KNNClassifier will chose the majority vote, while the KNNRegressor will chose the mean.
        """
        print("!!!")
        pass

    def _predict_one(self, input):
        """
        Predict class of one input.

        Arguments:
            input (np.array): (m)-dimensional array of point to be predicted.

        Returns:
            pred (int / str / ...): Predicted class.
        """
        if input.shape[0] != self.m:  # Check for compatible arrays
            message = f"Predicted point dimensions must be equal to training observations. "
            message += f"Was input: {input.shape[0]} and train: {self.m}. "
            raise ValueError(message)

        distances = np.zeros(self.n)
        for i in range(self.n):  # Calculate all the distances
            distances[i] = self._dist_func(input, self.x_train[i])

        nearest_labels = np.zeros(self.k)  # The labels of the k nearest neighbours
        sorted = distances.copy()  # The k first elements are the k smallest distances
        sorted.sort()  # Shortest distances
        for i in range(self.k):  # Find the k nearest neighbours
            min_index = np.where(distances == sorted[i])[0][0]  # Index of the i'th closest neighbour
            nearest_labels[i] = self.y_train[min_index]
        return self._choose_from_neighbours(nearest_labels)

    def predict(self, inputs):
        """
        Predicts on a array of inputs.

        Arguments:
            inputs (np.array): (c x m)-dimensional array of c inputs with m features,
                which will have classes predicted.

        Returns:
            predictions (np.array): (c)-dimensional array of the predicted classes.
        """
        if len(inputs.shape) == 1:  # If only one input vector, convert to (1 x m) array.
            inputs = np.expand_dims(inputs, 0)
        inputs = self._scaler(inputs, find_params=False, **self._scaling_params)  # Scale data

        predictions = np.zeros(len(inputs))
        for i in range(len(inputs)):
            predictions[i] = self._predict_one(inputs[i])

        return predictions


class KNNClassifier(KNN):
    """
    Class implementation of k nearest neighbour classifier.
    When predicting, finds the k nearest neighbours in the training set,
    and chooses the majority class of those neighbours as the prediction.
    Most of the methods are inherited from KNN.
    """
    def _choose_from_neighbours(self, neighbours):
        """
        Function for returning the predicted value from the k-nearest neighbours.
        Since this is classification, we chose the majority vote (mode).

        Arguments:
            neighbours (np.array): (k)-dimensional array of the k nearest neighbours.

        Returns:
            predidiction (int / string / ...): The predicted class from the majority vote.
        """
        return self._majority_vote(neighbours)

    def _majority_vote(self, input):
        """
        Returns the majority vote of an input. That is, the element
        that is repeated the most in the array "input". In case of a tie, the
        first appearing element in the tie is chosen.

        Arguments:
            input (np.array): (m)-dimensional array of classes (ints, str, etc).

        Returns:
            mode (int / str / ...): The element appearing the most in "input".
        """
        freq_dict = {}
        for element in input:  # Count all the occorencies
            if element in freq_dict:
                freq_dict[element] += 1
            else:  # Element not already in dict
                freq_dict[element] = 1

        # Find the max
        max_number = 0
        max_element = None
        for key, value in freq_dict.items():
            if value > max_number:  # Found a more frequent element
                max_number = value
                max_element = key

        return max_element


class KNNRegressor(KNN):
    """
    Class implementation of k nearest neighbour regressor.
    When predicting, finds the k nearest neighbours in the training set,
    and chooses the mean of those neighbours as the prediction.
    Most of the methods are inherited from KNN.
    """
    def _choose_from_neighbours(self, neighbours):
        """
        Function for returning the predicted value from the k-nearest neighbours.
        Since this is Regression, we will chose the mean.

        Arguments:
            neighbours (np.array): (k)-dimensional array of the k nearest neighbours.

        Returns:
            predidiction (int / string / ...): The predicted class from the majority vote.
        """
        return neighbours.mean()


if __name__ == "__main__":
    pass
