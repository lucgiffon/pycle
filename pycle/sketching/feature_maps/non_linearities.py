"""
Common sketching nonlinearities for the feature map.
"""

import numpy as np
import torch


def _complexExponentialTorch(t, T=2 * np.pi):
    """
    Evaluate the complex exponential at t.

    Parameters
    ----------
    t
        Sample position.
    T
        Period of the signal.
    Returns
    -------
        The complex exponential evaluation at t.
    """
    return torch.exp(1j * (2 * np.pi) * t / T)


# dict of nonlinearities for torch (no gradient necessary because autodiff)
_dico_nonlinearities_torch = {
    "complexexponential": _complexExponentialTorch,
    "none": lambda x: x,
    # "universalquantization": _universalQuantization,
    # "universalquantization": (lambda x: torch.sign(torch.cos(x)), None),
    # "universalquantization_complex": _universalQuantization_complex,
    # "universalquantization_complex": (lambda x: torch.sign(torch.cos(x)) + 1.j * torch.sign(torch.sin(x)), None),
    "cosine": lambda x: torch.cos(x),
    "identity": lambda x: x
}

# this dictionary is here for legacy purposes: necessary for assymmetric compressive learning.
_dico_normalization_rpf = {
    "complexexponential": 1,
    "none": 1,
    # "universalquantization": 2 / np.pi,
    # "universalquantization_complex": 4 / np.pi,
    "cosine": 1,
    "identity": 1
}