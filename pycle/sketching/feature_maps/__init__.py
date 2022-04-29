"""
Common sketch nonlinearities and derivatives
"""
import numpy as np
import torch
from loguru import logger


# cleaning add documentation
# cleaning check if use_torch parameter is necessary

def _complexExponential(t, T=2 * np.pi, use_torch=False):
    if use_torch:
        outer_exp = torch.exp
    else:
        outer_exp = np.exp

    return outer_exp(1j * (2 * np.pi) * t / T)


def _complexExponentialTorch(t, T=2 * np.pi):
    return _complexExponential(t, T, use_torch=True)


def _complexExponential_grad(t, T=2 * np.pi):
    return ((1j * 2 * np.pi) / T) * np.exp(1j * (2 * np.pi) * t / T)


def _universalQuantization(t, Delta=np.pi, centering=True, use_torch=False):
    if use_torch:
        backend = torch
    else:
        backend = np
    _div = t / Delta
    if (_div % 1 == 0).any():
        logger.warning(f"Input exactly multiple of {Delta} can lead to unexpected result in _universalQuantization")
    # t -= Delta/2
    if centering:
        # return ((t // Delta) % 2) * 2 - 1  # // stands for "int division
        # return ((torch.round(_div) + 1) % 2) * 2 - 1  # // stands for "int division
        return (backend.floor(_div - 0.5) % 2) * 2 - 1  # // stands for "int division
    else:
        return (backend.floor(_div - 0.5) % 2)  # centering=false => quantization is between 0 and +1


def _universalQuantization_complex(t, Delta=np.pi, centering=True, use_torch=False):
    return _universalQuantization(t, Delta=Delta, centering=centering, use_torch=use_torch) + 1.j * _universalQuantization(
        t - Delta/2, Delta=Delta, centering=centering, use_torch=use_torch)


def _sawtoothWave(t, T=2 * np.pi, centering=True):
    if centering:
        return (t % T) / T * 2 - 1
    else:
        return (t % T) / T  # centering=false => output is between 0 and +1


def _sawtoothWave_complex(t, T=2 * np.pi, centering=True):
    return _sawtoothWave(t - T / 4, T=T, centering=centering) + 1j * _sawtoothWave(t - T / 2, T=T, centering=centering)


def _triangleWave(t, T=2 * np.pi):
    return (2 * (t % T) / T) - (4 * (t % T) / T - 2) * ((t // T) % 2) - 1


def _fourierSeriesEvaluate(t, coefficients, T=2 * np.pi):
    """T = period
    coefficients = F_{-K}, ... , F_{-1}, F_{0}, F_{1}, ... F_{+K}"""
    K = (coefficients.shape[0] - 1) / 2
    ks = np.arange(-K, K + 1)
    # Pre-alloc
    ft = np.zeros(t.shape) + 0j
    for i in range(2 * int(K) + 1):
        ft += coefficients[i] * np.exp(1j * (2 * np.pi) * ks[i] * t / T)
    return ft

# dict of nonlinearities for torch (no gradient provided because autodiff)
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
_dico_normalization_rpf = {
    "complexexponential": 1,
    "none": 1,
    "universalquantization": 2 / np.pi,
    "universalquantization_complex": 4 / np.pi,
    "cosine": 1,
    "identity": 1
}