"""
Contains error and exceptions.
"""


class NotFittedError(ValueError, AttributeError):
    """
    Raise when a class is used before fitting.
    """

class NotTrainedError(ValueError, AttributeError):
    """
    Raise when model has not been trained yet.
    """