"""
File for visualizing pictures form the datasets
"""

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from common import check_arrays


def plot_mnist_random(images, preds=None, labels=None, transformed=False, show=True, n_random=10, n_cols=5):
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
        n_cols (int): The amount of images in each columns
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
            image = np.reshape(image, (28, 28))  # Reshape to original image size
        ax.imshow(image, cmap="gray")  # Plot the image

        title = ""  # Construct title
        if preds is not None:
            title += f"P: {preds[indices[i]]}"
        if preds is not None and labels is not None:
            title += ", "
        if labels is not None:
            title += f"Y: {labels[indices[i]]}"

        ax.title.set_text(title)

        plt.xticks(np.array([]))  # Remove ticks
        plt.yticks(np.array([]))

    plt.tight_layout()
    if show:
        plt.show()
    rc("text", color="black")  # Set text back to black


def plot_mnist_mislabeled(images, preds, labels, transformed=False, show=True, n_random=10, n_cols=5):
    """
    Plot random mislabeled images from the MNIST data given.

    Arguments:
        images (np.array): [n x 784] or [n x 28 x 28] array of mnist images.
        preds (np.array): [n] array of predicted labels corresponding to the images.
        labels (np.array): [n] array of labels corresponding to the images.
        tranformed (bool): If False, will reshape from 784 to 28 x 28.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
    """
    check_arrays(preds, labels, same_shape=True)
    indices = (preds != labels)
    wrong_images = images[indices]  # Index wrongly labeled images
    preds = preds[indices]
    labels = labels[indices]
    plot_mnist_random(wrong_images, preds, labels, transformed, show, n_random, n_cols)  # Use random plotting function


def plot_mnist_single(image, pred=None, label=None, transformed=False, show=True):
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
    """
    if not transformed:
        image = np.reshape(image, (28, 28))  # Reshape to original image size
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plt.imshow(image, cmap="gray")
    plt.xticks(np.array([]))  # Remove ticks
    plt.yticks(np.array([]))
    title = ""  # Construct title
    if pred is not None:
        title += f"Prediction: {pred}"
    if pred is not None and label is not None:
        title += ", "
    if label is not None:
        title += f"True label: {label}"
    plt.title(title)
    if show:
        plt.show()
