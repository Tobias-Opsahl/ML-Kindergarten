"""
File containing michalencious functions that are used in the models.
"""
import numpy as np
import matplotlib.pyplot as plt


class Distance:
    """
    Class for keeping distance functions (metrics).
    """
    @staticmethod
    def distanceL1(inputs1, inputs2):
        """
        Calculate the L1 distance of all the rows in inputs1 and inputs2. The inputs must be of the
        same lengths. This is defined as the sum over the absolute distances of each dimension.
        This is also called "manhattan distance".

        Arguments:
            inputs1 (np.array): (n x m)-dimensional array of n inputs with m features, containg the first points.
            inputs2 (np.array): (n x m)-dimensional array of n inputs with m features, containing the second points.

        Out:
            distances (np.array): (n) array of the distances computed (pairwise), with respect to the L1-distance.
        """
        if len(inputs1.shape) == 1:  # Turn (m) dimensional vector into (1 x m) dimensional array
            inputs1 = np.expand_dims(inputs1, 0)
        if len(inputs2.shape) == 1:
            inputs2 = np.expand_dims(inputs2, 0)
        check_arrays(inputs1, inputs2, dims=[0, 1])  # Check that the arrays are of same size.

        distances = np.absolute(inputs1 - inputs2).sum(axis=1)
        return distances

    @staticmethod
    def distanceL2(inputs1, inputs2):
        """
        Calculate the L2 distance of all the inputs in inputs1 and inputs2. The inputs must be of the
        same lengths. This is defined as the squareroot of the sum of the squared
        distances of each dimenion.
        This is also called Euclidian distance

        Arguments:
            inputs1 (np.array): (n x m)-dimensional array of n inputs with m features, containg the first points.
            inputs2 (np.array): (n x m)-dimensional array of n inputs with m features, containing the second points.

        Out:
            distances (np.array): (n) array of the distances computed (pairwise), with respect to the L1-distance.
        """
        if len(inputs1.shape) == 1:  # Turn (m) dimensional vector into (1 x m) dimensional array
            inputs1 = np.expand_dims(inputs1, 0)
        if len(inputs2.shape) == 1:
            inputs2 = np.expand_dims(inputs2, 0)
        check_arrays(inputs1, inputs2, dims=[0, 1])  # Check that the arrays are of same size.

        distances = np.square(inputs1 - inputs2).sum(axis=1)
        return distances


def check_array(x_array, check_type=True, check_shape=False, check_nans=True, check_inf=True):
    """
    Checks if array satisfies wanted properties.
    If they do not, raises ValueError.

    Arguments:
        x_array (np.array): The array to be checked.
        check_type (bool): Check if array is of type np.ndarray.
        check_shape (int): Checks that the length of the shape of the array is equal to "check_shape".
        check_nans (bool): Check if array contain nans.
        check_inf (bool): Check if array contains inf.
    """
    if check_type and not isinstance(x_array, np.ndarray):
        message = f"Argument must be of type np.ndarray. Was of type {type(x_array)}"
        raise ValueError(message)

    if check_shape and len(x_array.shape) != check_shape:
        message = f"Array expected to have {check_shape} dimensions, but had {len(x_array.shape)}."
        raise ValueError(message)

    if check_nans and np.isnan(x_array).any():
        message = f"Array contains nans."
        raise ValueError(message)

    if check_inf and np.isinf(x_array).any():
        message = f"Array contains nans."
        raise ValueError(message)


def check_arrays(x_array, y_array, dims=[0], check_type=True, check_shape=False, check_nans=False, check_inf=False):
    """
    Check if arrays has the correct shapes and types.

    Arguments:
        x_array (np.array): The first array to be checked.
        y_array (np.array): The second array to be checked.
        dims (list): List of ints of the dimensions that has to match in the arrays.
        check_type (bool): Check if array is of type np.ndarray.
        check_shape (int): Checks that the length of the shape of the array is equal to "check_shape".
        check_nans (bool): Check if array contain nans.
        check_inf (bool): Check if array contains inf.
    """
    check_array(x_array, check_type, check_shape, check_nans, check_inf)
    check_array(y_array, check_type, check_shape, check_nans, check_inf)
    for dim in dims:
        if x_array.shape[dim] != y_array.shape[dim]:
            message = f"Array dimenions mismatch. Dimenion {dim} must be of same length, "
            message += f"but was {x_array.shape[dim]} and {y_array.shape[dim]}. "
            message += f"Total shape was {x_array.shape} and {y_array.shape}."
            raise ValueError(message)


def integer_one_hot_encode(x_array):
    """
    One hot encodes x_array.
    This assumes that x_arrays only has integer, and that the max element + 1 is the amount of class.
    Therefore, the classes should probably have consequtive values from 0 to c - 1.

    Arguments:
        x_array (np.array): (n) array of values to be one-hot-encoded.

    Returns:
        one_hot_array (np.array): (n x c) array of one-hot-encoded data.
    """
    one_hot_array = np.zeros((x_array.shape[0], x_array.max() + 1))  # Initialize empty array
    one_hot_array[np.arange(x_array.shape[0]), x_array] = 1  # Index rows (arange) and columns (x_array)
    return one_hot_array


def find_accuracy(predictions, targets):
    """
    Calculates the accuracy of some predictions given its label.
    This is defined as (number of correct guesses) / (number of guesses).
    This works only for classification, because a predictions is only correct
    if they are compared equal.

    Arguments:
        predictions (np.array): (n)-dimensional array of predictions.
        targets (np.array): (n)-dimensional array of true targets
            for the same inputs as the predictions

    Returns:
        accuracy (float): The accuracy for the predictions
    """
    check_arrays(predictions, targets, dims=[0], check_type=True)

    accuracy = (predictions == targets).mean()
    return accuracy


def find_mse(predictions, targets):
    """
    Calculates Mean Squared Error (MSE).
    It is defined as 1/N sum_i (y_i - pred_i).

    Arguments:
        predictions (np.array): (n) array of predictions
        targets (np.array): (n) array of true target values

    Returns:
        mse (float): The mse calculated.
    """
    check_arrays(predictions, targets, dims=[0], check_type=True)

    mse = np.square(predictions - targets).mean()
    return mse


def plot_2d_blobs(x_blobs, y_blobs=None, title=None):
    """
    Plots 2d blobs.

    Arguments:
        x_blobs (np.array): (n x 2) dimensional input array
        y_blobs (np.array): (n) dimensional target array
        title (str): If not None, creates title of plot
    """
    if y_blobs is None:  # If no labels are provided, make all labels 0.
        y_blobs = np.zeros(len(x_blobs))
    plt.scatter(x_blobs[:, 0], x_blobs[:, 1], c=y_blobs)
    if title is not None:
        plt.title(title)


if __name__ == "__main__":
    # TODO: Move one-hot-encoder into here
    # TODO: Make a one-hot-encoder of numpy data, that also checks for already one-hot-econding,
    pass
