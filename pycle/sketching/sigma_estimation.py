"""
These funtions allows to select Sigma.

Sigma is the scale of the frequencies used to build the sketching operator.
It is a critical hyper-parameter for compressive clustering.

Beware: functions in this module work with numpy arrays.
"""
import warnings
from typing import NoReturn

import numpy as np
import scipy.optimize
import torch
from matplotlib import pyplot as plt

from pycle.sketching import computeSketch
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.frequency_sampling import drawFrequencies


# todo make all of this torch (create separate module and assert torch version give same result than numpy version)
def _fun_grad_fit_sigmas(p: np.ndarray, R: np.ndarray, z: np.ndarray):
    """
    Function and gradient to solve the optimization problem
        min_{w,sigs2} sum_{i = 1}^n ( z[i] - sum_{k=1}^K w[k]*exp(-R[i]^2*sig2[k]/2) )^2

    Parameters:
    -----------
        - p, a (2K,) numpy array obtained by stacking
            - w : (K,) numpy array corresponding to the weights of the mixture
            - sigs2 : (K,) numpy array the standard deviation of each Gaussian in the mixture
        - R: (n,) numpy array, data to fit (x label) The norms of the frequencies
        - z: (n,) numpy array, data to fit (y label) The corresponding sketch absolute value

    References:
    -----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.


    Returns:
    --------
        - The function evaluation
        - The gradient
    """

    K = p.size // 2
    w = p[:K]
    sigs2 = p[K:]
    n = R.size

    fun = 0
    grad = np.zeros(2 * K)
    for i in range(n):
        # todo something better than that naive implementation with the for loop
        fun += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) ** 2
        grad[:K] += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (- np.exp(-(sigs2 * R[i] ** 2) / 2.))  # grad of w
        grad[K:] += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (- w * np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (
                    -0.5 * R[i] ** 2)  # grad of sigma2

    return fun, grad


def _callback_fit_sigmas(p: np.ndarray) -> NoReturn:
    """
    Make the weights of the mixture of Gaussians (first half of p) sum to one, so that it is indeed a mixture.

    Parameters
    ----------
    p
        The parameters of the mixture (weights, sigmas)
    """
    K = p.size // 2
    p[:K] /= np.sum(p[:K])


def plot_sigma_estimation(z_sorted, frequencies_norm_sorted, z_tofit, R_tofit, number_of_gaussians_K, weights_of_gaussians, sigma_of_gaussians):
    """
    Plot the graph of the current step of sigma estimation.

    Parameters
    ----------
    z_sorted
        Sketch coefficients sorted by frequencies norm.
    frequencies_norm_sorted
        Frequencies norm sorted.
    z_tofit
        Only the extremum z coefficients in each bin.
    R_tofit
        The frequencies norm corresponding the z_tofit.
    number_of_gaussians_K
        The number of components in the mixture.
    weights_of_gaussians
        The weight of each component in the mixture.
    sigma_of_gaussians
        The parameter of the gaussian (otherwise centered) in the mixture.
    """
    plt.figure(figsize=(10, 5))
    rfit = np.linspace(0, frequencies_norm_sorted.max(), 100)
    zfit = np.zeros(rfit.shape)
    for k in range(number_of_gaussians_K):
        zfit += weights_of_gaussians[k] * np.exp(-(sigma_of_gaussians[k] * rfit ** 2) / 2.)
    plt.plot(frequencies_norm_sorted, np.abs(z_sorted), '.')
    plt.plot(R_tofit, z_tofit, '.')
    plt.plot(rfit, zfit)
    plt.xlabel('R')
    plt.ylabel('|z|')
    plt.show()


def estimate_Sigma_from_sketch(z: np.ndarray, Phi: MatrixFeatureMap, K=1, c=20, mode='max', sigma2_bar=None,
                               weights_bar=None, should_plot=False):
    """
    Estimate the right sigmas by fitting a mixture of Gaussians to the extrememums-by-bin coefficients in the sketch.

    The procedure is the following:

    - Sort the frequencies (Phi.Omega);
    - Partition with respect to amplitude of the frequencies. This gives `c` bins.;
    - Keep one value in each corresponding bin in the sketch being the `mode` value. (max or min)
    - Fit a mixture of gaussian distributions to these values.
    - Return the sigma squared (variance) of that gaussian.

    All Gaussians in the mixture are centered.

    Parameters
    ----------
    z:
        The sketch obtained from `Phi`.
    Phi:
        Sketching operator. Necessary attributes are:
        - Omega : the matrix of frequencies,
        - c_norm: the sketch normalization constant
        - m: the number of frequencies
    K:
        The number of gaussians parameterized by sigma in the mixture.
    c:
        Number of bins where to sample the data to fit to the exponential.
    mode:
        "min" or "max" to salect the sample representing each bin
    sigma2_bar:
        The initial sigmas. If None, it is sampled in uniform(0.3, 1.6, K).
    weights_bar:
        The weights of the mixture
    should_plot
        Tells to plot the result of the optimisation scheme.

    References
    ----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.

    Returns
    -------
        The array of each sigmas in the mixture.
    """

    # Parse
    if mode == 'max':
        mode_criterion = np.argmax
    elif mode == 'min':
        mode_criterion = np.argmin
    else:
        raise ValueError("Unrecocgnized mode ({})".format(mode))

    # sort the freqs by norm
    frequencies_norm = np.linalg.norm(Phi.Omega.cpu().numpy(), axis=0)
    i_sort = np.argsort(frequencies_norm)
    frequencies_norm = frequencies_norm[i_sort]
    z_sort = z[i_sort] / Phi.c_norm  # sort and normalize individual entries to 1

    # Initialization
    # number of freqs per box
    s = Phi.m // c
    # init point
    if sigma2_bar is None:
        sigma2_bar = np.random.uniform(0.3, 1.6, K)
    if weights_bar is None:
        weights_bar = np.ones(K) / K

    # find the indices of the max of each block
    indices_samples = np.empty(c)
    for ic in range(c):
        j_max = mode_criterion(np.abs(z_sort)[ic * s:(ic + 1) * s]) + ic * s
        indices_samples[ic] = j_max
    indices_samples = indices_samples.astype(int)
    R_tofit = frequencies_norm[indices_samples]
    z_tofit = np.abs(z_sort)[indices_samples]

    # Set up the fitting opt. problem
    def f(_p):
        return _fun_grad_fit_sigmas(_p, R_tofit, z_tofit)  # cost

    p0 = np.zeros(2 * K)  # initial point
    p0[:K] = weights_bar  # w -> these are the weights of the gaussian mixture. Must sum to one.
    p0[K:] = sigma2_bar

    # Bounds of the optimization problem
    bounds = []
    # bounds for the weigths
    for k in range(K):
        bounds.append([1e-5, 1])
    # bounds for the sigmas -> cant change too much
    for k in range(K):
        bounds.append([5e-4 * sigma2_bar[k], 2e3 * sigma2_bar[k]])

    # Solve the sigma^2 optimization problem
    sol = scipy.optimize.minimize(f, p0, jac=True, bounds=bounds, callback=_callback_fit_sigmas)
    p = sol.x
    weights_bar = np.array(p[:K]) / np.sum(p[:K])
    sigma2_bar = np.array(p[K:])

    # Plot if required
    if should_plot:
        plot_sigma_estimation(
            z_sorted=z_sort,
            frequencies_norm_sorted=frequencies_norm,
            z_tofit=z_tofit,
            R_tofit=R_tofit,
            number_of_gaussians_K=K,
            weights_of_gaussians=weights_bar,
            sigma_of_gaussians=sigma2_bar
        )

    return sigma2_bar


def estimate_Sigma(dataset: np.ndarray, m0, K=None, c=20, n0=None, drawFreq_type="AR", nIterations=5, mode='max',
                   verbose=0, device="cpu"):
    """Automatically estimates the "Sigma" parameter(s) (the scale of data clusters) for generating the sketch operator.

    We assume here that Sigma = sigma2_bar * identity matrix.
    To estimate sigma2_bar, lightweight sketches of size m0 are generated from (a small subset of) the dataset
    with candidate values for sigma2_bar. Then, sigma2_bar is updated by fitting a Gaussian
    to the absolute values of the obtained sketch.

    Arguments:
        - dataset: (n,d) numpy array, the dataset X: n examples in dimension d
        - m0: int, number of candidate 'frequencies' to draw (can be typically smaller than m).
        - K:  int (default 1), number of scales to fit (if > 1 we fit a scale mixture)
        - c:  int (default 20), number of 'boxes' (i.e. number of maxima of sketch absolute values to fit)
        - n0: int or None, if given, n0 samples from the dataset are subsampled to be used for Sigma estimation
        - drawType: a string indicating the sampling pattern (Lambda) to use in the pre-sketches, either:
            -- "gaussian"       or "G"  : Gaussian sampling > Lambda = N(0,Sigma^{-1})
            -- "foldedGaussian" or "FG" : Folded Gaussian sampling (i.e., the radius is Gaussian)
            -- "adaptedRadius"  or "AR" : Adapted Radius heuristic
        - nIterations: int (default 5), the maximum number of iteration (typically stable after 2 iterations)
        - mode: 'max' (default) or 'min', describe which sketch entries per block to fit
        - verbose: 0,1 or 2. Number of plots of the sigma estimation process.
            0: no plot; 1: only last plot; 2: all the plots.

    References:
    -----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.

    Returns: If K = 1:
                - Sigma: (d,d)-numpy array, the (diagonal) estimated covariance of the clusters in the dataset;
             If K > 1: a tuple (w,Sigma) representing the scale mixture model, where:
                - Sigma: (K,d,d)-numpy array, the dxd covariances in the scale mixture
                (uniformity is assumed for mixture weights)
    """

    return_format_is_matrix = K is None
    K = 1 if K is None else K

    (n, d) = dataset.shape
    # X is the subsampled dataset containing only n0 examples
    if n0 is not None and n0 < n:
        if 0 < n0 <= 1:
            n0 = int(n0 * n)
        X = dataset[np.random.choice(n, n0, replace=False)]
    else:
        X = dataset

    # Check if we dont overfit the empirical Fourier measurements
    if m0 < (K * 2) * c:
        warnings.warn("WARNING: overfitting regime detected for frequency sampling fitting")

    # Initialization
    # maxNorm = np.max(np.linalg.norm(X,axis=1))
    sigma2_bar = np.random.uniform(0.3, 1.6, K)
    weights_bar = np.ones(K) / K

    X = torch.from_numpy(X).to(torch.device(device))
    # Actual algorithm
    for i in range(nIterations):
        # Draw frequencies according to current estimate
        sigma2_bar_matrix = np.outer(sigma2_bar, np.eye(d)).reshape(K, d, d)  # covariances in (K,d,d) format
        Omega0 = drawFrequencies(drawFreq_type, d, m0, Sigma=(weights_bar, sigma2_bar_matrix), return_torch=True)

        # Compute unnormalized complex exponential sketch
        Phi0 = MatrixFeatureMap("complexexponential", Omega0, device=torch.device(device))
        z0 = computeSketch(X, Phi0)

        should_plot = verbose > 1 or (verbose > 0 and i >= nIterations - 1)
        sigma2_bar = estimate_Sigma_from_sketch(z0.cpu().numpy(), Phi0,
                                                K, c, mode, sigma2_bar, weights_bar, should_plot)

    if return_format_is_matrix:
        Sigma = sigma2_bar[0] * np.eye(d)
    else:
        sigma2_bar_matrix = np.outer(sigma2_bar, np.eye(d)).reshape(K, d, d)  # covariances in (K,d,d) format
        Sigma = sigma2_bar_matrix

    return Sigma
