import numpy as np
import pickle
import gzip
# from scaling import NormalScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')
from scaling import StandardScaler


def load_mnist(path="data/", transform=True, normalize=False):
    """
    https://yann.lecun.com/exdb/mnist/ (do not open in chrome for some reason)
    Load the MNIST dataset. It should be saved as mnist.pkl.gz in the same folder.

    Arguments:
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        transform (bool): If True, scales pixels down to be between [0, 1].
        normalize (bool): If True, normalizes the dataset. Will normalize the
            validation and test set according to the scaling parameters of the train set.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784]
            is the array of all the training images, and y_train [50000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    return _load_mnist_generic("mnist", path, transform, normalize)


def load_fashion_mnist(path="data/", transform=True, normalize=False):
    """
    https://github.com/zalandoresearch/fashion-mnist
    Load the fashion MNIST dataset. It should be saved as mnist.pkl.gz in the same folder.
    The data is also split with a validation-set, taking 10000 of the 60000 train pictures
    into a validation set, making it on the same format as the original MNIST dataset.

    Arguments:
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        transform (bool): If True, scales pixels down to be between [0, 1].
        normalize (bool): If True, normalizes the dataset. Will normalize the
            validation and test set according to the scaling parameters of the train set.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784]
            is the array of all the training images, and y_train [50000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    return _load_mnist_generic("fashion-mnist", path, transform, normalize)


def _load_mnist_generic(dataset, path="data/", transform=True, normalize=False):
    """
    Generic function for reading either the original MNIST dataset of fashion-MNIST dataset.

    Arguments:
        dataset (str): The dataset to read, either "mnist" or "fashion-mnist".
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        transform (bool): If True, scales pixels down to be between [0, 1].
        normalize (bool): If True, normalizes the dataset. Will normalize the
            validation and test set according to the scaling parameters of the train set.

    Returns:
        training_data (tuple): Tuple consisting of (x_train, y_train), where x_train [50000 x 784]
            is the array of all the training images, and y_train [50000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        validation_data (tuple): Tuple consisting of (x_val, y_val), where x_val [10000 x 784]
            is the array of all the validation images, and y_val [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
        test_data (tuple): Tuple consisting of (x_test, y_test), where x_test [10000 x 784]
            is the array of all the testing images, and y_test [10000] is the array of true targets,
            not one-hot-encoded, but in digits 0-9.
    """
    with gzip.open(path + dataset + "/train-labels-idx1-ubyte.gz", "rb") as infile:
        train_labels = np.frombuffer(infile.read(), dtype=np.uint8, offset=8)
    n_train = len(train_labels)
    with gzip.open(path + dataset + "/train-images-idx3-ubyte.gz", "rb") as infile:
        train_data = np.frombuffer(infile.read(), dtype=np.uint8, offset=16).reshape(n_train, 784)

    with gzip.open(path + dataset + "/t10k-labels-idx1-ubyte.gz", "rb") as infile:
        test_labels = np.frombuffer(infile.read(), dtype=np.uint8, offset=8)
    n_test = len(test_labels)
    with gzip.open(path + dataset + "/t10k-images-idx3-ubyte.gz", "rb") as infile:
        test_data = np.frombuffer(infile.read(), dtype=np.uint8, offset=16).reshape(n_test, 784)

    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=10000)
    x_test, y_test = test_data, test_labels

    if transform:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        x_train = x_train / 255
        x_val = x_val / 255
        x_test = x_test / 255

    if normalize:  # Standard normal scaling.
        normal_scaler = StandardScaler()
        x_train = normal_scaler.fit_transform(x_train, axis=None)
        x_val = normal_scaler.transform(x_val)  # Validation and test uses same parameters as train.
        x_test = normal_scaler.transform(x_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    pass
    # from IPython import embed
    # a, b, c = load_fashion_mnist(path="", transform=False, normalize=False)
    # A, B, C = load_mnist(path="", transform=False, normalize=False)
    # embed()
