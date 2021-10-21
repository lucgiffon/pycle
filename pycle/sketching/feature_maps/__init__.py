"""
Common sketch nonlinearities and derivatives
"""
import numpy as np
import torch


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


def _universalQuantization(t, Delta=np.pi, centering=True):
    if centering:
        return ((t // Delta) % 2) * 2 - 1  # // stands for "int division
    else:
        return ((t // Delta) % 2)  # centering=false => quantization is between 0 and +1


def _universalQuantization_complex(t, Delta=np.pi, centering=True):
    return _universalQuantization(t - Delta / 2, Delta=Delta, centering=centering) + 1j * _universalQuantization(
        t - Delta, Delta=Delta, centering=centering)


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


# dict of nonlinearities and their gradient returned as a tuple
_dico_nonlinearities = {
    "complexexponential": (_complexExponential, _complexExponential_grad),
    "universalquantization": (_universalQuantization, None),
    "universalquantization_complex": (_universalQuantization_complex, None),
    "sawtooth": (_sawtoothWave, None),
    "sawtooth_complex": (_sawtoothWave_complex, None),
    "cosine": (lambda x: np.cos(x), lambda x: -np.sin(x)),
    "none": (None, None)
}

# dict of nonlinearities for torch (no gradient provided because autodiff)
_dico_nonlinearities_torch = {
    "complexexponential": (_complexExponentialTorch, None),
    "none": (None, None)
}