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


def find_binary_cross_entropy(predictions, targets):
    """
    Returns binary cross entropy loss, also called the log loss.
    Should be binary encoded, one float for each predictions, and targets 0 or 1.

    Arguments:
        predictions (np.array): (n) array of predictions.
        targets (np.array): (n) array of true target values.

    Returns:
        cross_entropy (float): The cross_entropy calculated.
    """
    check_arrays(predictions, targets, dims=[0])
    return - np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))


def find_multiclass_cross_entropy(predictions, targets):
    """
    Returns multiclass cross entropy loss, also called the log loss
    Predictions should be (c) sized probabilities and targets should be one-hot-encoded

    Arguments:
        predictions (np.array): (n x c) array of predictions
        targets (np.array): (n x c) array of true target values

    Returns:
        cross_entropy (float): The cross_entropy calculated.
    """
    check_arrays(predictions, targets, dims=[0, 1])
    return - np.mean((targets * np.log(predictions)).sum(axis=1))


def sigmoid(x_array):
    """
    Performs the sigmoid function on every element in x_array.

    Arguments:
        x_array (np.array): (n) array of float values.

    Returns:
        simoids (np.array): (n) array of the sigmoid function, every element is between 0 and 1.
    """
    return 1 / (1 + np.exp(-x_array))


def softmax(x_array):
    """
    Performs the softmax function on x_array.

    Arguments:
        x_array (np.arary): [n x c] array of n inpits and c classes.

    Returns:
        softmaxes (np.array): [n x c] array, where every row are positive, sums to one, and one c grows
            monotone with respect to an increase in the same c from x_arary (properties of softmax).
    """

    denominator = np.exp(x_array).sum(axis=1)
    softmaxes = np.exp(x_array) / np.expand_dims(denominator, axis=1)
    return softmaxes


def plot_decision_regions(x_data, y_data, classifier, size=None, n_points=500,
                          title="Decision Regions", feature1="x1", feature2="x2", show=True):
    """
    Plots the decision regions for the classes predicted by the classifier "classifier",
    along with the data "x_data" with labels "y_data".
    For this to work, the data (x_data) must have exactly two features, which will each be
    represented as one dimensions in the plane. y_data should not be one-hot-encoded, and
    will be represented for a color for each point.

    Arguments:
        x_data (np.array): (n x 2) array of input data (must have exactly two features).
        y_data (np.array): (n) array of labels (not one-hot encoded)
        classifier (classifier): Trained instance of a classifier class.
            Must have a "predict()" method implemented
        n_points (int): Amount of points to be plotted along each axis.
        size (tuple): Tuple of two floats indicating size of plot
        title (str): Title of plot.
        feature1 (str): Name of first x-feature.
        feature2 (str): Name of second x-feature.
        show (boolean): If True, calls plt.show()
    """
    # Find borders for the plot
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    # Make a meshgrid to predict on.
    x_points = np.linspace(x_min, x_max, n_points)  # Shape [n_points]
    y_points = np.linspace(y_min, y_max, n_points)  # Shape n_points
    # Make meshgrid:
    # x_mesh: [n_points x n_points], x_points are copied n_points time as rows (each row is one x_points
    # y_mesh: [n_points x n_points], y_points are copied n_points time as columns (each column is one y_ponts)
    x_mesh, y_mesh = np.meshgrid(x_points, y_points)
    # Pairs the inputs of the flattened mesh_grids. predict_data: [n_points * n_points x 2]
    predict_data = np.c_[x_mesh.ravel(), y_mesh.ravel()]  # Flatten and pair
    preds = classifier.predict(predict_data)
    preds_mesh = preds.reshape(x_mesh.shape)  # Reshape back to [n_points x n_points]
    plt.figure(figsize=size)
    plt.contourf(x_mesh, y_mesh, preds_mesh, alpha=0.2, cmap="Pastel1")  # Plot decision regions
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, s=10, cmap="Paired")  # Plot points

    # Plot stuff
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    if show:
        plt.show()


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
    pass
