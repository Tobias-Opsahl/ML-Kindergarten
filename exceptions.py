"""
Contains error and exceptions.
"""


class NotFittedError(ValueError, AttributeError):
    """
    Raise when a class is used before training / fitting.
    """
