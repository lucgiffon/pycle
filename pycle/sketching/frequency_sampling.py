"""
Frequency sampling functions
"""
import numpy as np
from matplotlib import pyplot as plt
import torch

def drawDithering(m, bounds=None):
    """Draws m samples a <= x < b, with bounds=(a,b) (default: (0,2*pi))."""
    if bounds is None:
        (lowb, highb) = (0, 2 * np.pi)
    else:
        (lowb, highb) = bounds
    return np.random.uniform(low=lowb, high=highb, size=m)


def drawFrequencies_Gaussian(d, m, Sigma=None, randn_mat_0_1=None, seed=None):
    """draws frequencies according to some sampling pattern"""


    if Sigma is None:
        Sigma = np.identity(d)

    if randn_mat_0_1 is None:
        Om = np.random.RandomState(seed).multivariate_normal(np.zeros(d), np.linalg.inv(Sigma), m).T  # inverse of sigma
    else:
        assert randn_mat_0_1.shape == (d, m)
        Om = np.linalg.inv(Sigma) @ randn_mat_0_1
    return Om


def drawFrequencies_FoldedGaussian(d, m, Sigma=None, randn_mat_0_1=None, seed=None):
    """draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from folded Gaussian with variance 1, phi uniform"""
    if Sigma is None:
        Sigma = np.identity(d)
    R = np.abs(np.random.RandomState(seed).randn(m))  # folded standard normal distribution radii
    if randn_mat_0_1 is None:
        phi = np.random.RandomState(seed).randn(d, m)
    else:
        phi = randn_mat_0_1
    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma))

    Om = SigFact @ phi * R

    return Om


def sampleFromPDF(pdf, x, nsamples=1, seed=None):
    """x is a vector (the support of the pdf), pdf is the values of pdf eval at x"""
    # Note that this can be more general than just the adapted radius distribution

    pdf = pdf / np.sum(pdf)  # ensure pdf is normalized

    cdf = np.cumsum(pdf)

    # necessary?
    cdf[-1] = 1.

    sampleCdf = np.random.RandomState(seed).uniform(0, 1, nsamples)

    sampleX = np.interp(sampleCdf, cdf, x)

    return sampleX


def pdfAdaptedRadius(r, KMeans=False):
    """up to a constant"""
    if KMeans:
        return r * np.exp(-(r ** 2) / 2)  # Dont take the gradient according to sigma into account
    else:
        return np.sqrt(r ** 2 + (r ** 4) / 4) * np.exp(-(r ** 2) / 2)


def drawFrequencies_AdaptedRadius(d, m, Sigma=None, KMeans=False, randn_mat_0_1=None, seed=None):
    """draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform"""
    if Sigma is None:
        Sigma = np.identity(d)

    # Sample the radii
    r = np.linspace(0, 5, 2001)
    R = sampleFromPDF(pdfAdaptedRadius(r, KMeans), r, nsamples=m, seed=seed)

    if randn_mat_0_1 is None:
        phi = np.random.randn(d, m)
    else:
        phi = randn_mat_0_1
    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma))

    Om = SigFact @ phi * R

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


def drawFrequencies(drawType, d, m, Sigma=None, nb_cat_per_dim=None, randn_mat_0_1=None, seed=None, use_torch=False):
    """Draw the 'frequencies' or projection matrix Omega for sketching.

    Arguments:
        - drawType: a string indicating the sampling pattern (Lambda) to use, one of the following:
            -- "gaussian"       or "G"  : Gaussian sampling > Lambda = N(0,Sigma^{-1})
            -- "foldedGaussian" or "FG" : Folded Gaussian sampling (i.e., the radius is Gaussian)
            -- "adaptedRadius"  or "AR" : Adapted Radius heuristic
        - d: int, dimension of the data to sketch
        - m: int, number of 'frequencies' to draw (the target sketch dimension)
        - Sigma: is either:
            -- (d,d)-numpy array, the covariance of the data (note that we typically use Sigma^{-1} in the frequency domain).
            -- a tuple (w,cov) describing a scale mixture of Gaussians where,
                -- w:  (K,)-numpy array, the weights of the scale mixture
                -- cov: (K,d,d)-numpy array, the K different covariances in the mixture
            -- None: same as Sigma = identity matrix (belongs to (d,d)-numpy array case)
                 If Sigma is None (default), we assume that data was normalized s.t. Sigma = identity.
        - nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous. Frequencies for int data is drawn as follows:
                    1. Chose one dimension among the categorical ones, set Omega along all others to zero
                    2. For the chosen dimension with C categories, we draw its component omega ~ U({0,...,C-1}) * 2*pi/C

    Returns:
        - Omega: (d,m)-numpy array containing the 'frequency' projection matrix
    """
    # Parse drawType input
    if drawType.lower() in ["drawfrequencies_gaussian", "gaussian", "g"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_Gaussian(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed)
    elif drawType.lower() in ["drawfrequencies_foldedgaussian", "foldedgaussian", "folded_gaussian", "fg"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_FoldedGaussian(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed)
    elif drawType.lower() in ["drawfrequencies_adapted", "adaptedradius", "adapted_radius", "ar"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_AdaptedRadius(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed)
    elif drawType.lower() in ["drawfrequencies_adapted_kmeans", "adaptedradius_kmeans", "adapted_radius_kmeans", "arkm",
                              "ar-km"]:
        drawFunc = lambda _a, _b, _c: drawFrequencies_AdaptedRadius(_a, _b, _c, KMeans=True, randn_mat_0_1=randn_mat_0_1, seed=seed)
    else:
        raise ValueError("drawType not recognized")

    # Handle no input
    if Sigma is None:
        Sigma = np.identity(d)

    # Handle
    if isinstance(Sigma, np.ndarray):
        Omega = drawFunc(d, m, Sigma)

    # Handle mixture-type input
    elif isinstance(Sigma, tuple):
        (w, cov) = Sigma  # unpack
        K = w.size
        # Assign the frequencies to the mixture components
        assignations = np.random.choice(K, m, p=w)
        Omega = np.zeros((d, m))
        for k in range(K):
            active_index = (assignations == k)
            if any(active_index):
                Omega[:, np.where(active_index)[0]] = drawFunc(d, active_index.sum(), cov[k])

    elif (isinstance(Sigma, float) or isinstance(Sigma, int)) and Sigma > 0:
        Omega = drawFunc(d, m, Sigma * np.eye(d))

    else:
        raise ValueError("Sigma not recognized")

    # If needed, overwrite the integer entries
    if nb_cat_per_dim is not None:
        intg_index = np.nonzero(nb_cat_per_dim)[0]
        d_intg = np.size(intg_index)

        Omega_intg = np.zeros((d_intg, m))
        for intgdim_localindex, intg_globalindex in enumerate(intg_index):
            C = nb_cat_per_dim[intg_globalindex]
            Omega_intg[intgdim_localindex, :] = (2 * np.pi / C) * np.random.randint(0, C, (1, m))
        # Mask
        masks_pool = np.eye(d_intg)
        masks = masks_pool[np.random.choice(d_intg, m)].T
        Omega_intg = Omega_intg * masks

        Omega[intg_index] = Omega_intg

    if use_torch:
        return torch.from_numpy(Omega)
    else:
        return Omega


def multi_scale_frequency_sampling(dim, m, scale_min, scale_max, nb_scales, sampling_method):
    scales = np.logspace(scale_min, scale_max, num=nb_scales)
    Omega = np.zeros((dim, m))
    size_each_scale = m // nb_scales
    remaining_frequencies = m % nb_scales
    index_frequency = 0
    for sigma in scales:
        # choose the number of frequencies to sample.
        # if m is not dividable by nb_scales, there is a remaining to distribute amond the first frequencies sampled.
        if remaining_frequencies > 0:
            nb_frequencies_scale = size_each_scale + 1  # +1 to distribute the frequencies
            remaining_frequencies -= 1
        else:
            nb_frequencies_scale = size_each_scale

        frequencies_sigma = drawFrequencies(sampling_method, dim, nb_frequencies_scale, sigma * np.eye(dim))

        next_index_frequency = index_frequency + nb_frequencies_scale
        Omega[:, index_frequency:next_index_frequency] = frequencies_sigma
        index_frequency = next_index_frequency

    return Omega


if __name__ == "__main__":
    om = multi_scale_frequency_sampling(10, 20, -4, 0, 5, "arkm")
    print(om.shape)