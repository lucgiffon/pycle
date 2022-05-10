from typing import Any
from pycle.utils.metrics import loglikelihood_GMM


class SingletonMeta(type):
    """
    Implements Singleton design pattern.

    Use like:

    ```
    class MyClass(meta=SingletonMeta):
        ...
    ```
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        When cls is "called" (asked to be initialised), return a new object (instance of cls) only if no object of that cls
        have already been created. Else, return the already existing instance of cls.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
            Existing instance or new instance of cls.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def is_number(possible_number: Any) -> bool:
    """
    Check if input argument is a number like a float or integer.

    Return False if it is a string or even a np.ndarray with only one element in it.

    Parameters
    ----------
    possible_number

    Returns
    -------
        True if input is actually a number.
    """
    has_len = hasattr(possible_number, "__len__")
    if not has_len:
        return True
    else:
        try:
            len(possible_number)
        except:
            return True

    return False
