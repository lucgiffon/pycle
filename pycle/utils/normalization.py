"""
This module contains possible normalization functions to apply to the dataset.

It has been function :func:`linf_normalization` that has produced the best results in my experience.
"""
from typing import Union, Literal
import numpy as np
import torch


def get_backend_from_object(obj: Union[torch.Tensor, np.ndarray]):
    """
    Returns the backend handling the input object.

    torch if obj is a torch.Tensor, numpy otherwise.

    Parameters
    ----------
    obj
        Numpy array or torch tensor.

    Returns
    -------

    """
    if isinstance(obj, torch.Tensor):
        return torch
    else:
        return np


def get_normalization_factor_from_string(
        X: Union[torch.Tensor, np.ndarray],
        normalization_string: Literal['l_2-unit-ball', 'l_inf-unit-ball']) -> Union[torch.Tensor, np.ndarray]:
    """
    Returns the proper normalization factor for the given input matrix and the asked normalization.

    If the asked normalization is:

    - 'l_inf-unit-ball', then returns the max value in the array. \
    Divide the dataset by this value to have it all contained in the unit sphere
    - 'l_2-unit-ball', then returns the max sample norm. \
    Divide the dataset by this value to have the maximum sample norm equal to one.

    Parameters
    ----------
    X
        (N x D) The data matrix to normalize. N: the dataset size, D: the number of features.
    normalization_string
        One of ('l_2-unit-ball', 'l_inf-unit-ball') to specify the kind of normalization

    Returns
    -------
    The asked normalization factor.
    """

    backend = get_backend_from_object(X)

    if normalization_string in ['l_2-unit-ball']:
        return backend.linalg.norm(X, axis=1).max() + 1e-6  # plus smth to not get a zero have
    elif normalization_string in ['l_inf-unit-ball']:
        return backend.abs(X).max() + 1e-6
    else:
        raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalization_string))
