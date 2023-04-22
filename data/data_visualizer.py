"""
File for visualizing pictures form the datasets
"""

from common import check_arrays
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')


def plot_image_single(image, pred=None, label=None, transformed=False, dataset="MNIST", show=True, title=None):
    """
    Display a single picture.

    Arguments:
        image (np.array): [p] or [h x w] array of mnist image.
        pred (int): Prediction corresponding to the image.
            If provided, will include the prediction in the title.
        label (int): label corresponding to the images. If given, will
            include label as the title to the image.
        tranformed (bool): If False, will reshape from p to h x w, according to
            the argument "dataset".
        dataset (str): The dataset to be plotted. This determines the reshape size, if
            "transformed" is False. Must be in ["MNIST", "CIFAR"].
        show (bool): If True, will call plt.show().
        title (str): Title for the plot.
    """
    if not transformed:
        if dataset.upper() == "MNIST":
            image = np.reshape(image, (28, 28))  # Reshape to original image size
        elif dataset.upper() == "CIFAR":
            image = np.reshape(image, (3, 32, 32)).transpose(1, 2, 0)
        else:
            message = f"Argument \"dataset\" must be in [\"MNIST\", \"CIFAR\"]. Was {dataset}. "
            raise ValueError(message)
    plt.imshow(image, cmap="gray")
    plt.xticks(np.array([]))  # Remove ticks
    plt.yticks(np.array([]))
    if title is None:
        title = ""  # Construct title
    else:
        title += ". "
    if pred is not None:
        title += f"Prediction: {pred}"
    if pred is not None and label is not None:
        title += ", "
    if label is not None:
        title += f"True label: {label}"
    plt.title(title)
    if show:
        plt.show()


def plot_images_random(images, preds=None, labels=None, transformed=False,
                       dataset="MNIST", show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random images of the data given.

    Arguments:
        images (np.array): [n x p] or [n x h x w] array of images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n] array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        tranformed (bool): If False, will reshape accodring to "dataset"
        dataset (str): The dataset to be plotted. This determines the reshape size, if
            "transformed" is False. Must be in ["MNIST", "CIFAR"].
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    fig = plt.figure()
    n = images.shape[0]
    indices = np.random.choice(n, n_random, replace=False)  # Chose indices for random images to plot
    n_rows = int(np.ceil(n_random / n_cols))
    for i in range(n_random):
        if preds is not None and labels is not None:  # Make title blue in wrong predictions
            if preds[indices[i]] != labels[indices[i]]:
                rc("text", color="blue")
            else:
                rc("text", color="black")

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        image = images[indices[i]]
        if not transformed:
            if dataset.upper() == "MNIST":
                image = np.reshape(image, (28, 28))  # Reshape to original image size
            elif dataset.upper() == "CIFAR":
                image = np.reshape(image, (3, 32, 32)).transpose(1, 2, 0)
            else:
                message = f"Argument \"dataset\" must be in [\"MNIST\", \"CIFAR\"]. Was {dataset}. "
                raise ValueError(message)
        ax.imshow(image, cmap="gray")  # Plot the image

        sub_title = ""  # Construct sub_title
        if preds is not None:
            sub_title += f"P: {preds[indices[i]]}"
        if preds is not None and labels is not None:
            sub_title += ", "
        if labels is not None:
            sub_title += f"Y: {labels[indices[i]]}"

        ax.title.set_text(sub_title)

        plt.xticks(np.array([]))  # Remove ticks
        plt.yticks(np.array([]))

    rc("text", color="black")  # Set text back to black
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_images_mislabeled(images, preds, labels, transformed=False, dataset="MNIST",
                           show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random mislabeled images from the data provided.

    Arguments:
        images (np.array): [n x p] or [n x h x w] array of images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from c to h x w, depending on
            the argument "dataset".
        dataset (str): The dataset to be plotted. This determines the reshape size, if
            "transformed" is False. Must be in ["MNIST", "CIFAR"].
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    check_arrays(preds, labels, same_shape=True)
    indices = (preds != labels)
    wrong_images = images[indices]  # Index wrongly labeled images
    preds = preds[indices]
    labels = labels[indices]
    # Use random plotting function
    plot_images_random(wrong_images, preds, labels, transformed, dataset, show, n_random, n_cols, title)


def plot_images_worst(images, logits, labels, transformed=False, dataset="MNIST",
                      show=True, n_images=10, n_cols=5, title=None):
    """
    Plot (probably mislabeled) images that corresponds to the worst predictions. This means
    the value for the true class and the predicted logit value is as different as possible.

    Arguments:
        images (np.array): [n x p] or [n x h x w] array of images.
        logits (np.array): [n] array of predicted logits values (either softmax or other activation function outputs).
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from c to h x w, depending on
            the argument "dataset".
        dataset (str): The dataset to be plotted. This determines the reshape size, if
            "transformed" is False. Must be in ["MNIST", "CIFAR"].
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    predicted_logits = logits[np.arange(len(labels)), labels]
    indices = predicted_logits.argsort()
    wrong_images = images[indices[:n_images]]
    logits = logits[indices[:n_images]]
    labels = labels[indices[:n_images]]
    preds = logits.argmax(axis=1)
    plot_images_random(wrong_images, preds, labels, transformed, dataset, show, n_images, n_cols, title)


def plot_mnist_single(image, pred=None, label=None, transformed=False, show=True, title=None):
    """
    Display a single picture from the MNIST dataset.

    Arguments:
        image (np.array): [784] or [28 x 28] array of mnist images.
        pred (int): Prediction corresponding to the image.
            If provided, will include the prediction in the title.
        label (int): label corresponding to the images. If given, will
            include label as the title to the image.
        tranformed (bool): If False, will reshape from 784 to 28 x 28.
        show (bool): If True, will call plt.show().
        title (str): Title for the plot.
    """
    plot_image_single(image, pred, label, transformed, "MNIST", show, title)


def plot_mnist_random(images, preds=None, labels=None, transformed=False, show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random images from the MNIST data given.

    Arguments:
        images (np.array): [n x 784] or [n x 28 x 28] array of mnist images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n] array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        tranformed (bool): If False, will reshape from 784 to 28 x 28.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    plot_images_random(images, preds, labels, transformed, "MNIST", show, n_random, n_cols, title)


def plot_mnist_mislabeled(images, preds, labels, transformed=False, show=True, n_random=10, n_cols=5, title=None):
    """
    Plot random mislabeled images from the MNIST data given.

    Arguments:
        images (np.array): [n x 784] or [n x 28 x 28] array of mnist images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from 784 to 28 x 28.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    plot_images_mislabeled(images, preds, labels, transformed, "MNIST", show, n_random, n_cols, title)


def plot_mnist_worst(images, logits, labels, transformed=False, show=True, n_images=10, n_cols=5, title=None):
    """
    Plot (probably mislabeled) images that corresponds to the worst predictions. This means
    the value for the true class and the predicted logit value is as different as possible.

    Arguments:
        images (np.array): [n x 784] or [n x 28 x 28] array of mnist images.
        logits (np.array): [n] array of predicted logits values (either softmax or other activation function outputs).
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from 784 to 28 x 28.
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    plot_images_worst(images, logits, labels, transformed, "MNIST", show, n_images, n_cols, title)


def plot_cifar_single(image, pred=None, label=None, transformed=False, show=True, title=None):
    """
    Display a single picture from the CIFAR dataset.

    Arguments:
        image (np.array): [3072] or [32 x 32 x 3] array of a CIFAR image.
        pred (int): Prediction corresponding to the image.
            If provided, will include the prediction in the title.
        label (int): label corresponding to the images. If given, will
            include label as the title to the image.
        tranformed (bool): If False, will reshape from [3072] or [32 x 32 x 3]
        show (bool): If True, will call plt.show().
        title (str): Title for the plot.
    """
    plot_image_single(image, pred, label, transformed, "CIFAR", show, title)


def plot_cifar_random(images, preds=None, labels=None, transformed=False, show=True, n_random=10, n_cols=5, title=None):
    """
    Display a single picture from the CIFAR dataset.

    Arguments:
        image (np.array): [n x 3072] or [n x 32 x 32 x 3] array of a CIFAR image
        preds (np.array): [n] array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n] array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        tranformed (bool): If False, will reshape from [n x 3072] or [n x 32 x 32 x 3]
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    plot_images_random(images, preds, labels, transformed, "CIFAR", show, n_random, n_cols, title)


def plot_cifar_mislabeled(images, preds, labels, transformed=False, show=True, n_random=10, n_cols=5, title=None):
    """
    Display a single picture from the CIFAR dataset.

    Arguments:
        image (np.array): [n x 3072] or [n x 32 x 32 x 3] array of a CIFAR images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from [n x 3072] or [n x 32 x 32 x 3]
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    plot_images_mislabeled(images, preds, labels, transformed, "CIFAR", show, n_random, n_cols, title)


def plot_cifar_worst(images, logits, labels, transformed=False, show=True, n_images=10, n_cols=5, title=None):
    """
    Plot (probably mislabeled) images that corresponds to the worst predictions. This means
    the value for the true class and the predicted logit value is as different as possible.

    Arguments:
        image (np.array): [n x 3072] or [n x 32 x 32 x 3] array of a CIFAR images.
        logits (np.array): [n] array of predicted logits values (either softmax or other activation function outputs).
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from [n x 3072] or [n x 32 x 32 x 3]
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
    """
    plot_images_worst(images, logits, labels, transformed, "CIFAR", show, n_images, n_cols, title)
