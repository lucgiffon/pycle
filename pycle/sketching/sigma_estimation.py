"""
funtions allows to estimate Sigma (some utils first, main function is "estimate_Sigma")
"""
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from pycle.sketching import computeSketch
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.frequency_sampling import drawFrequencies


def _fun_grad_fit_sigmas(p, R, z):
    """
    Function and gradient to solve the optimization problem
        min_{w,sigs2} sum_{i = 1}^n ( z[i] - sum_{k=1}^K w[k]*exp(-R[i]^2*sig2[k]/2) )^2
    Arguments:
        - p, a (2K,) numpy array obtained by stacking
            - w : (K,) numpy array
            - sigs2 : (K,) numpy array
        - R: (n,) numpy array, data to fit (x label)
        - z: (n,) numpy array, data to fit (y label)
    Returns:
        - The function evaluation
        - The gradient
    """

    K = p.size // 2
    w = p[:K]
    sigs2 = p[K:]
    n = R.size
    # Naive implementation, schellekensvTODO better?
    fun = 0
    grad = np.zeros(2 * K)
    for i in range(n):
        fun += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) ** 2
        grad[:K] += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (- np.exp(-(sigs2 * R[i] ** 2) / 2.))  # grad of w
        grad[K:] += (z[i] - w @ np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (- w * np.exp(-(sigs2 * R[i] ** 2) / 2.)) * (
                    -0.5 * R[i] ** 2)  # grad of sigma2
    return (fun, grad)


def _callback_fit_sigmas(p):
    K = p.size // 2
    p[:K] /= np.sum(p[:K])


def estimate_Sigma_from_sketch(z, Phi, K=1, c=20, mode='max', sigma2_bar=None, weights_bar=None, should_plot=False):
    # Parse
    if mode == 'max':
        mode_criterion = np.argmax
    elif mode == 'min':
        mode_criterion = np.argmin
    else:
        raise ValueError("Unrecocgnized mode ({})".format(mode))

    # sort the freqs by norm
    Rs = np.linalg.norm(Phi.Omega, axis=0)
    i_sort = np.argsort(Rs)
    Rs = Rs[i_sort]
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
    jqs = np.empty(c)
    for ic in range(c):
        j_max = mode_criterion(np.abs(z_sort)[ic * s:(ic + 1) * s]) + ic * s
        jqs[ic] = j_max
    jqs = jqs.astype(int)
    R_tofit = Rs[jqs]
    z_tofit = np.abs(z_sort)[jqs]

    # Set up the fitting opt. problem
    f = lambda p: _fun_grad_fit_sigmas(p, R_tofit, z_tofit)  # cost

    p0 = np.zeros(2 * K)  # initial point
    p0[:K] = weights_bar  # w
    p0[K:] = sigma2_bar

    # Bounds of the optimization problem
    bounds = []
    for k in range(K): bounds.append([1e-5, 1])  # bounds for the weigths
    for k in range(K): bounds.append(
        [5e-4 * sigma2_bar[k], 2e3 * sigma2_bar[k]])  # bounds for the sigmas -> cant cange too much

    # Solve the sigma^2 optimization problem
    sol = scipy.optimize.minimize(f, p0, jac=True, bounds=bounds, callback=_callback_fit_sigmas)
    p = sol.x
    weights_bar = np.array(p[:K]) / np.sum(p[:K])
    sigma2_bar = np.array(p[K:])

    # Plot if required
    if should_plot:
        plt.figure(figsize=(10, 5))
        rfit = np.linspace(0, Rs.max(), 100)
        zfit = np.zeros(rfit.shape)
        for k in range(K):
            zfit += weights_bar[k] * np.exp(-(sigma2_bar[k] * rfit ** 2) / 2.)
        plt.plot(Rs, np.abs(z_sort), '.')
        plt.plot(R_tofit, z_tofit, '.')
        plt.plot(rfit, zfit)
        plt.xlabel('R')
        plt.ylabel('|z|')
        plt.show()

    return sigma2_bar


def estimate_Sigma(dataset, m0, K=None, c=20, n0=None, drawFreq_type="AR", nIterations=5, mode='max', verbose=0):
    """Automatically estimates the "Sigma" parameter(s) (the scale of data clusters) for generating the sketch operator.

    We assume here that Sigma = sigma2_bar * identity matrix.
    To estimate sigma2_bar, lightweight sketches of size m0 are generated from (a small subset of) the dataset
    with candidate values for sigma2_bar. Then, sigma2_bar is updated by fitting a Gaussian
    to the absolute values of the obtained sketch. Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.

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
        - verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.

    Returns: If K = 1:
                - Sigma: (d,d)-numpy array, the (diagonal) estimated covariance of the clusters in the dataset;
             If K > 1: a tuple (w,Sigma) representing the scale mixture model, where:
                - w:     (K,)-numpy array, the weigths of the scale mixture (sum to 1)
                - Sigma: (K,d,d)-numpy array, the dxd covariances in the scale mixture
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
    if (m0 < (K * 2) * c):
        print("WARNING: overfitting regime detected for frequency sampling fitting")

    # Initialization
    # maxNorm = np.max(np.linalg.norm(X,axis=1))
    sigma2_bar = np.random.uniform(0.3, 1.6, K)
    weights_bar = np.ones(K) / K

    # Actual algorithm
    for i in range(nIterations):
        # Draw frequencies according to current estimate
        sigma2_bar_matrix = np.outer(sigma2_bar, np.eye(d)).reshape(K, d, d)  # covariances in (K,d,d) format
        Omega0 = drawFrequencies(drawFreq_type, d, m0, Sigma=(weights_bar, sigma2_bar_matrix))

        # Compute unnormalized complex exponential sketch
        Phi0 = MatrixFeatureMap("ComplexExponential", Omega0)
        z0 = computeSketch(X, Phi0)

        should_plot = verbose > 1 or (verbose > 0 and i >= nIterations - 1)
        sigma2_bar = estimate_Sigma_from_sketch(z0, Phi0, K, c, mode, sigma2_bar, weights_bar, should_plot)

    if return_format_is_matrix:
        Sigma = sigma2_bar[0] * np.eye(d)
    else:
        sigma2_bar_matrix = np.outer(sigma2_bar, np.eye(d)).reshape(K, d, d)  # covariances in (K,d,d) format
        Sigma = (weights_bar, sigma2_bar_matrix)

    return Sigma