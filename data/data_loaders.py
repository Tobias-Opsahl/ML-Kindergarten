from scaling import StandardScaler
import numpy as np
import pickle
import gzip
# from scaling import NormalScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')


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


def load_cifar10(path="data/", transform=True, normalize=False, validation_set=True):
    """
    Loads the CIFAR10 dataset.

    Arguments:
        dataset (str): The dataset to read, either "mnist" or "fashion-mnist".
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        transform (bool): If True, scales pixels down to be between [0, 1].
        normalize (bool): If True, normalizes the dataset. Will normalize the
            validation and test set according to the scaling parameters of the train set.
        validation_set (bool): If True, will return a validation set based on 10000 images from the
            training data.

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
    x_train = np.zeros((50000, 3072), dtype=np.uint8)
    y_train = np.zeros(50000, dtype=np.uint8)
    for i, batch_number in enumerate(["1", "2", "3", "4", "5"]):  # The training data is saved in five batch-numbers
        filename = path + "cifar/" + "cifar-10-batches-py/data_batch_" + batch_number
        with open(filename, "rb") as infile:
            dict = pickle.load(infile, encoding="bytes")
            x_train[i * 10000: (i + 1) * 10000] = dict[b'data']
            y_train[i * 10000: (i + 1) * 10000] = dict[b'labels']
    filename = path + "cifar/" + "cifar-10-batches-py/test_batch"
    x_test = np.zeros((10000, 3072), dtype=np.uint8)
    y_test = np.zeros(10000, dtype=np.uint8)
    with open(filename, "rb") as infile:
        dict = pickle.load(infile, encoding="bytes")
        x_test[:] = dict[b'data']
        y_test[:] = dict[b'labels']

    if transform:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        x_train = x_train / 255
        x_test = x_test / 255

    if normalize:  # Standard normal scaling.
        normal_scaler = StandardScaler()
        x_train = normal_scaler.fit_transform(x_train, axis=None)
        x_test = normal_scaler.transform(x_test)

    if validation_set:  # Split training set into training and validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    return (x_train, y_train), (x_test, y_test)


def load_cifar100(path="data/", transform=True, normalize=False, validation_set=True, labels="fine_labels"):
    """
    Loads the CIFAR100 dataset.

    Arguments:
        dataset (str): The dataset to read, either "mnist" or "fashion-mnist".
        path (str): Path to enter the folder where the data is store, relative to
            where this function is called.
        transform (bool): If True, scales pixels down to be between [0, 1].
        normalize (bool): If True, normalizes the dataset. Will normalize the
            validation and test set according to the scaling parameters of the train set.
        validation_set (bool): If True, will return a validation set based on 10000 images from the
            training data.
        labels (str): Either "fine_labels" for the 100 categories, or "coarse_labels" for 20 categories.

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
    # Fix labels arguments
    if labels not in ["fine_labels", "fine", "coarse_labels", "coarse"]:
        message = f"Argument \"labels\" must be in [\"fine_labels\", \"coarse_labels\"]. Was {labels}. "
        raise ValueError(message)
    if labels == "fine_labels" or labels == "fine":
        labels = b'fine_labels'
    elif labels == "coarse_labels" or labels == "coarse":
        labels = b'coarse_labels'

    x_train = np.zeros((50000, 3072), dtype=np.uint8)
    y_train = np.zeros(50000, dtype=np.uint8)
    filename = path + "cifar/" + "cifar-100-python/train"
    with open(filename, "rb") as infile:
        dict = pickle.load(infile, encoding="bytes")
        x_train[:] = dict[b'data']
        y_train[:] = dict[labels]
    filename = path + "cifar/" + "cifar-100-python/test"
    x_test = np.zeros((10000, 3072), dtype=np.uint8)
    y_test = np.zeros(10000, dtype=np.uint8)
    with open(filename, "rb") as infile:
        dict = pickle.load(infile, encoding="bytes")
        x_test[:] = dict[b'data']
        y_test[:] = dict[labels]

    if transform:  # Pixels are between 0 and 255, so simply dividing by 255 turn pixels into [0, 1]
        x_train = x_train / 255
        x_test = x_test / 255

    if normalize:  # Standard normal scaling.
        normal_scaler = StandardScaler()
        x_train = normal_scaler.fit_transform(x_train, axis=None)
        x_test = normal_scaler.transform(x_test)

    if validation_set:  # Split training set into training and validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    pass
