"""
Common sketching nonlinearities for the feature map.
"""

import numpy as np
import torch
from loguru import logger


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
def _universalQuantization(t, Delta=np.pi, centering=True):
    """
    Return the real universal quantization (UQ) activation function, that is: sign(cos(x)) if delta is pi.

    Note: _universalQuantization(t - pi/2) = sign(sin(x))

    Parameters
    ----------
    t
        Where to evaluate the (UQ) function
    Delta
        The quantization bin size, that is, the half period of changing sign (default: 2 pi / 2 = pi) .
    centering
        Tells to center the result, that is UQ(0) lands in {-1, 1} instead of {0, 1}.

    References
    ----------

    https://arxiv.org/pdf/2104.10061.pdf


    Returns
    -------
        The universal quantization evaluated at t.
    """
    _div = t / Delta
    if (_div % 1 == 0).any():
        logger.warning(f"Input exactly multiple of {Delta} can lead to unexpected result in _universalQuantization")
    if centering:
        # return ((t // Delta) % 2) * 2 - 1  # // stands for "int division
        # return ((torch.round(_div) + 1) % 2) * 2 - 1  # // stands for "int division
        return (torch.floor(_div - 0.5) % 2) * 2 - 1  # // stands for "int division
    else:
        return torch.floor(_div - 0.5) % 2  # centering=false => quantization is between 0 and +1


def _universalQuantization_complex(t, Delta=np.pi, centering=True):
    """
    Return the complex universal quantization (UQ) activation function, that is: sign(cos(x)) + i sign(sin(x)).


    Parameters
    ----------
    t
        Where to evaluate the (UQ) function
    Delta
        The quantization bin size, that is, the half period of changing sign (default: 2 pi / 2 = pi) .
    centering
        Tells to center the result, that is UQ(0) lands in {-1, 1} instead of {0, 1}.

    References
    ----------

    https://arxiv.org/pdf/2104.10061.pdf


    Returns
    -------
        The complex universal quantization evaluated at t.
    """
    # cleaning test it
    return _universalQuantization(t, Delta=Delta, centering=centering) + 1.j * _universalQuantization(
        t - Delta/2, Delta=Delta, centering=centering)


_dico_nonlinearities_torch = {
    "complexexponential": _complexExponentialTorch,
    "none": lambda x: x,
    "universalquantization": _universalQuantization,
    # "universalquantization": (lambda x: torch.sign(torch.cos(x)), None),
    "universalquantization_complex": _universalQuantization_complex,
    # "universalquantization_complex": (lambda x: torch.sign(torch.cos(x)) + 1.j * torch.sign(torch.sin(x)), None),
    "cosine": lambda x: torch.cos(x),
    "identity": lambda x: x
}

# this dictionary is here for legacy purposes: necessary for assymmetric compressive learning.
_dico_normalization_rpf = {
    "complexexponential": 1,
    "none": 1,
    "universalquantization": 2 / np.pi,
    "universalquantization_complex": 4 / np.pi,
    "cosine": 1,
    "identity": 1
}