import numpy as np
import torch
from matplotlib import pyplot as plt

from pycle.sketching.frequency_sampling import multi_scale_frequency_sampling, drawFrequencies, sampleFromPDF


def overproduce(dim, max_number_of_frequencies, overproduce_factor, strategy="MULTI_SCALE"):
    # cleaning add documentation
    # For this simple example, assume we have a priori a rough idea of the size of the clusters
    Sigma = 0.1 * np.eye(dim)
    # Pick the dimension m: 5*K*d is usually (just) enough in clustering (here m = 50)
    sketch_dim = overproduce_factor * max_number_of_frequencies

    if strategy == "MULTI_SCALE":
        Omega = multi_scale_frequency_sampling(dim, sketch_dim, -2, 0,
                                       10, "arkm",
                                       use_torch=True)
    elif strategy == "uniform":
        Omega = drawFrequencies_UniformRadius(dim, sketch_dim, 1e-2, 1e0, use_torch=True)

    else:
        Omega = drawFrequencies("FoldedGaussian", dim, sketch_dim, Sigma, return_torch=True)

    xi = torch.rand(sketch_dim) * np.pi * 2
    return Omega, xi


def choose(base_sketch, max_nb_freq, strategy="in-boundaries", threshold=0.01):
    # cleaning add documentation
    abs_base_sketch = base_sketch.abs()
    if strategy == "closest-to-mid":
        sorted_indices = torch.argsort((abs_base_sketch - 0.5).abs())
        return torch.sort(sorted_indices[:max_nb_freq])[0]
    else:
        indices_greater_than_min = threshold < abs_base_sketch
        indices_lower_than_max = abs_base_sketch < 1-threshold
        accepted_bool_indices = torch.logical_and(indices_lower_than_max, indices_greater_than_min)
        accepted_indices = torch.arange(len(base_sketch))[accepted_bool_indices]
        selected_indices = (torch.ones(len(accepted_indices)) / len(accepted_indices)).multinomial(num_samples=max_nb_freq, replacement=False)
        indices_ok = accepted_indices[selected_indices]
        return indices_ok


def drawFrequencies_UniformRadius(d, m, min_val, max_val, randn_mat_0_1=None, use_torch=False):
    """draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform"""
    from scipy.stats import loguniform

    R = loguniform.rvs(min_val, max_val, size=m)

    if randn_mat_0_1 is None:
        phi = np.random.randn(d, m)
    else:
        phi = randn_mat_0_1

    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere

    Om = phi * R

    # cleaning replace these use_torch with a parameter more explicit about the returned type
    if use_torch:
        return torch.from_numpy(Om)
    else:
        return Om


def pdf_diffOfGaussians(r, GMM_upper=None, GMM_lower=None):
    """Here, GMM is given in terms of SD and not variance"""
    if isinstance(GMM_upper, tuple):
        (weights_upper, sigmas_upper) = GMM_upper
    elif GMM_upper is None:
        weights_upper = np.array([])  # Empty array
    else:
        (weights_upper, sigmas_upper) = (np.array([1.]), np.array([GMM_upper]))

    if isinstance(GMM_lower, tuple):
        (weights_lower, sigmas_lower) = GMM_lower
    elif GMM_lower is None:
        weights_lower = np.array([])
    else:
        (weights_lower, sigmas_lower) = (np.array([1.]), np.array([GMM_lower]))

    res = np.zeros(r.shape)
    # Add
    for k in range(weights_upper.size):
        res += weights_upper[k] * np.exp(-0.5 * (r ** 2) / (sigmas_upper[k] ** 2))
    # Substract
    for k in range(weights_lower.size):
        res -= weights_lower[k] * np.exp(-0.5 * (r ** 2) / (sigmas_lower[k] ** 2))

    # Ensure pdf is positive
    pdf_is_negative = res < 0
    if any(pdf_is_negative):
        print(res[:5])
        # Print a warning if the negative pdf values are significant (not due to rounding errors)
        tol = 1e-8
        if np.max(np.abs(res[np.where(pdf_is_negative)[0]])) > tol:
            print("WARNING: negative pdf values detected and replaced by zero, check the validity of your input")
        # Correct the negative values
        res[np.where(pdf_is_negative)[0]] = 0.

    return res


def drawFrequencies_diffOfGaussians(d, m, GMM_upper, GMM_lower=None, verbose=0):
    """
    draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, schellekensvTODO, phi uniform
    """

    # reasonable sampling
    n_Rs = 1001
    if isinstance(GMM_upper, tuple):
        R_max = 4 * np.max(GMM_upper[1])  # GMM_upper is (weights, cov)-type tuple
    else:
        R_max = 4 * GMM_upper
    r = np.linspace(0, R_max, n_Rs)

    if verbose > 0:
        plt.plot(r, pdf_diffOfGaussians(r, GMM_upper, GMM_lower))
        plt.xlabel('frequency norm r')
        plt.ylabel('pdf(r)')
        plt.show()

    # sample from the diff of gaussians pdf
    R = sampleFromPDF(pdf_diffOfGaussians(r, GMM_upper, GMM_lower), r, nsamples=m)

    phi = np.random.randn(d, m)
    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere

    Om = phi * R

    return Om