import numpy as np
import torch
from loguru import logger


# cleaning add documentation for everything here
def _universalQuantization(t, Delta=np.pi, centering=True):
    _div = t / Delta
    if (_div % 1 == 0).any():
        logger.warning(f"Input exactly multiple of {Delta} can lead to unexpected result in _universalQuantization")
    # t -= Delta/2
    if centering:
        # return ((t // Delta) % 2) * 2 - 1  # // stands for "int division
        # return ((torch.round(_div) + 1) % 2) * 2 - 1  # // stands for "int division
        return (torch.floor(_div - 0.5) % 2) * 2 - 1  # // stands for "int division
    else:
        return (torch.floor(_div - 0.5) % 2)  # centering=false => quantization is between 0 and +1


def _universalQuantization_complex(t, Delta=np.pi, centering=True):
    # cleaning test it
    return _universalQuantization(t, Delta=Delta, centering=centering) + 1.j * _universalQuantization(
        t - Delta/2, Delta=Delta, centering=centering)


def _sawtoothWave(t, T=2 * np.pi, centering=True):
    # cleaning test it
    if centering:
        return (t % T) / T * 2 - 1
    else:
        return (t % T) / T  # centering=false => output is between 0 and +1


def _sawtoothWave_complex(t, T=2 * np.pi, centering=True):
    # cleaning test it
    return _sawtoothWave(t - T / 4, T=T, centering=centering) + 1j * _sawtoothWave(t - T / 2, T=T, centering=centering)


def _triangleWave(t, T=2 * np.pi):
    # cleaning test it
    return (2 * (t % T) / T) - (4 * (t % T) / T - 2) * ((t // T) % 2) - 1


def _fourierSeriesEvaluate(t, coefficients, T=2 * np.pi):
    """
    T = period
    coefficients = F_{-K}, ... , F_{-1}, F_{0}, F_{1}, ... F_{+K}
    """
    # cleaning test it
    K = (coefficients.shape[0] - 1) / 2
    ks = torch.arange(-K, K + 1)
    # Pre-alloc
    ft = torch.zeros(t.shape) + 0j
    for i in range(2 * int(K) + 1):
        ft += coefficients[i] * torch.exp(1j * (2 * np.pi) * ks[i] * t / T)
    return ft