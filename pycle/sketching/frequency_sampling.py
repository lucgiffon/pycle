"""
Frequency sampling functions.

These functions allow to find a frequencies matrix that will be used in the random fourier features map
for the sketching operator.
"""
import numpy as np
import torch

from pycle.legacy.frequency_sampling import multi_scale_frequency_sampling
from pycle.utils import is_number


def drawDithering(m, bounds=None):
    """
    Draws m samples a <= x < b, with bounds=(a,b) (default: (0,2*pi)).

    The dithering vector can be used as the `xi` parameter of feature maps in order to deal with asymetry between
    the sketching feature map and learning feature map.

    Parameters
    ----------
    m:
        The size of the dithering vector.
    bounds:
        The bounds of the uniform distribution where to sample the dithering.

    Returns
    -------
        The vector of dithering.
    """
    if bounds is None:
        (lowb, highb) = (0, 2 * np.pi)
    else:
        (lowb, highb) = bounds
    return np.random.uniform(low=lowb, high=highb, size=m)


def sampleFromPDF(pdf, x, nsamples=1, seed=None) -> np.ndarray:
    """
    pdf means  "probability density function".

    This function uses the Inverse transform sampling method to generate samples from the pdf.

    Note that this can be more general than just the adapted radius distribution

    Parameters
    ----------
    pdf:
        Vector containing the values of the pdf at x. eg: pdf[i] = pdf(x[i])
    x:
        Vector being the support of the pdf
    nsamples:
        Number of samples
    seed:
        Seed of the random generator.

    References
    ----------

    see wikipedia: inverse transform sampling

    Returns
    -------
        (nsamples, ) array of samples of the pdf.

    """
    pdf = pdf / np.sum(pdf)  # ensure pdf is normalized

    cdf = np.cumsum(pdf)
    assert np.isclose(cdf[-1], 1)
    cdf[-1] = 1.

    # the inverse cdf (implemented by the interp call below) applied on a uniform sample in 0, 1 gives a sample of the
    # base probability function (see wikipedia: inverse transform sampling)
    sampleCdf = np.random.RandomState(seed).uniform(0, 1, nsamples)
    sampleX = np.interp(sampleCdf, xp=cdf, fp=x)

    return sampleX


def pdfAdaptedRadius(r: np.ndarray, KMeans=False) -> np.ndarray:
    """
    Probability Density Function of the Adapted Radius distribution used to sample the radius of the frequencies
    in the sketching operator.

    This pdf is defined up to a constant.

    Parameters:
    -----------
    r:
        Vector of points where to estimate the pdf of the Adapted Radius distribution.

    References:
    -----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.1.

    Returns:
    --------
        Vector of evalution of the pdf AR at r values.

    """

    if KMeans:
        # Dont take the gradient according to sigma into account
        return r * np.exp(-(r ** 2) / 2)
    else:
        return np.sqrt(r ** 2 + (r ** 4) / 4) * np.exp(-(r ** 2) / 2)


def draw_radii(m: int, R_seeds, KMeans: bool) -> np.ndarray:
    """
    Sample the m-sized vectors of radii for each of the R_seeds.

    Parameters
    ----------
    m:
        size of the output array.
    R_seeds:
        Sequence of seeds to use for each vector
    KMeans:
        Tells if those radii will be used for Kmeans (true) or gaussian mixture estimation (false).


    Returns
    -------
        (len(R_seeds), m) 2D np.ndarray of the radii for each seed.
    """
    r = np.linspace(0, 5, 2001)
    lst_R = []
    for R_seed in R_seeds:
        # Sample the radii
        lst_R.append(sampleFromPDF(pdfAdaptedRadius(r, KMeans), r, nsamples=m, seed=R_seed))
    R = np.array(lst_R)
    return R


def drawFrequencies_Gaussian(d: int, m: int, Sigma=None, randn_mat_0_1=None, seed: int = None,
                             keep_splitted: bool = False):
    """
    Draws frequencies according to a Gaussian distribution with variance 1/sigma**2

    Parameters
    ----------
    d:
        The dimension of the Gaussian
    m:
        The number of frequencies to sample
    Sigma:
        The variance of the Gaussian. Can be a number of a square np.ndarray
    randn_mat_0_1:
        Use a pre-sampled Gaussian matrix with coefficients sampled in N(0, 1)
    seed:
        Seed for random number generation.
    keep_splitted:
        Return the frequencies matrix in the form of a tuple (sigma, directions, radii) where the frequency matrix
        equals: sigma @ (directions * radii)

    References
    ----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.1.

    Returns
    -------
        (d, m) np.ndarray the frequencies matrix if keep_splitted is false
        tuple (sigma, directions, radii) otherwise.
    """

    if Sigma is None:
        Sigma = np.identity(d)

    if keep_splitted:
        directions = randn_mat_0_1 or np.random.RandomState(seed).randn(d, m)
        radii = np.linalg.norm(directions, axis=0)
        directions /= radii
        if is_number(Sigma):
            Sigma = np.array([Sigma])
        return np.linalg.inv(Sigma), directions, radii
    else:
        if randn_mat_0_1 is None:
            Om = np.random.RandomState(seed).multivariate_normal(np.zeros(d), np.linalg.inv(Sigma), m).T
        else:
            assert randn_mat_0_1.shape == (d, m)

            if is_number(Sigma):
                Om = 1./Sigma * randn_mat_0_1
            else:
                Om = np.linalg.inv(Sigma) @ randn_mat_0_1

        return Om


def drawFrequencies_FoldedGaussian(d, m, Sigma=None, randn_mat_0_1=None, seed=None, keep_splitted=False, R_seeds=None):
    """
    Draws frequencies according to folded Gaussian distribution.

    omega = R*Sigma^{-1/2}*phi, for R from folded Gaussian with variance 1, phi uniform on the unit sphere

    Parameters
    ----------
    d:
        The dimension of the Gaussian
    m:
        The number of frequencies to sample
    Sigma:
        The variance of the Gaussian. Can be a number of a square np.ndarray
    randn_mat_0_1:
        Use a pre-sampled Gaussian matrix with coefficients sampled in N(0, 1)
    seed:
        Seed for random number generation.
    keep_splitted:
        Return the frequencies matrix in the form of a tuple (sigma, directions, radii) where the frequency matrix
        equals: sigma @ (directions * radii)
    R_seeds:
        Sequence of seeds to use for each vector

    References
    ----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.1.

    Returns
    -------
        (d, m) np.ndarray the frequencies matrix if keep_splitted is false
        tuple (sigma, directions, radii) otherwise.
    """

    if Sigma is None:
        Sigma = np.identity(d)

    if R_seeds is None:
        R_seeds = [seed]

    lst_R = []
    for R_seed in R_seeds:
        lst_R.append(np.abs(np.random.RandomState(R_seed).randn(m)))  # folded standard normal distribution radii
    R = np.squeeze(np.array(lst_R))

    if randn_mat_0_1 is None:
        phi = np.random.RandomState(seed).randn(d, m)
    else:
        phi = randn_mat_0_1

    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere

    if is_number(Sigma):
        SigFact = np.array([1. / np.sqrt(Sigma)])
    elif isinstance(Sigma, np.ndarray) and Sigma.ndim == 1:
        SigFact = 1. / np.sqrt(Sigma)
    else:
        SigFact = np.linalg.inv(np.linalg.cholesky(Sigma))

    if keep_splitted:
        return SigFact, phi, R.T
    else:
        if is_number(Sigma):
            Om = SigFact * phi * R
        else:
            Om = SigFact @ phi * R
        return Om


def drawFrequencies_AdaptedRadius(d, m, Sigma=None, KMeans=False, randn_mat_0_1=None, seed=None, keep_splitted=False,
                                  R_seeds=None):
    """
    Draws frequencies according to Adapted Radius distribution.

    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform on unit sphere


    Parameters
    ----------
    d:
        The dimension of the Gaussian
    m:
        The number of frequencies to sample
    Sigma:
        The variance of the Gaussian. Can be a number of a square np.ndarray
    KMeans:
        Tells if the frequencies are sampled for kmeans problem (True) or GMM estimation (False)
    randn_mat_0_1:
        Use a pre-sampled Gaussian matrix with coefficients sampled in N(0, 1)
    seed:
        Seed for random number generation.
    keep_splitted:
        Return the frequencies matrix in the form of a tuple (sigma, directions, radii) where the frequency matrix
        equals: sigma @ (directions * radii)
    R_seeds:
        Sequence of seeds to use for each vector

    References
    ----------

    Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.1.

    Returns
    -------
        (d, m) np.ndarray the frequencies matrix if keep_splitted is false
        tuple (sigma, directions, radii) otherwise.
    """

    if Sigma is None:
        Sigma = np.identity(d)

    if R_seeds is None:
        R_seeds = [seed]

    R = draw_radii(m, R_seeds, KMeans)

    if randn_mat_0_1 is None:
        phi = np.random.RandomState(seed).randn(d, m)
    else:
        phi = randn_mat_0_1
    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere

    if is_number(Sigma):
        SigFact = np.array([1. / np.sqrt(Sigma)])
    elif isinstance(Sigma, np.ndarray) and Sigma.ndim == 1:
        SigFact = 1. / np.sqrt(Sigma)
    else:
        SigFact = np.linalg.inv(np.linalg.cholesky(Sigma))

    if keep_splitted:
        return SigFact, phi, np.squeeze(R.T)
    else:
        if SigFact.ndim == 2:
            Om = SigFact @ (phi * R)
        else:
            Om = rebuild_Omega_from_sigma_direction_R(SigFact, phi, R.T, math_module=np)
        return Om


def drawFrequencies(drawType, d, m, Sigma=None, nb_cat_per_dim=None, randn_mat_0_1=None, seed=None, return_torch=False, keep_splitted=False, R_seeds=None):
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
            -- A number greater than zero: it will be treated like the identity times this number.
            -- A np.ndarray of numbers greater than zero: it will be treated like the concatenation of as many identities times these numbers.
        - nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous. Frequencies for int data is drawn as follows:
                    1. Chose one dimension among the categorical ones, set Omega along all others to zero
                    2. For the chosen dimension with C categories, we draw its component omega ~ U({0,...,C-1}) * 2*pi/C
        - randn_mat_0_1: np.ndarray, a random matrix with gaussian 0, 1 entries that is used as core for producing the frequencies.

    Returns:
        - Omega: (d,m)-numpy array containing the 'frequency' projection matrix
    """
    # Parse drawType input
    if drawType.lower() in ["drawfrequencies_gaussian", "gaussian", "g"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_Gaussian(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed, keep_splitted=keep_splitted)
    elif drawType.lower() in ["drawfrequencies_foldedgaussian", "foldedgaussian", "folded_gaussian", "fg"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_FoldedGaussian(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed, keep_splitted=keep_splitted, R_seeds=R_seeds)
    elif drawType.lower() in ["drawfrequencies_adapted", "adaptedradius", "adapted_radius", "ar"]:
        drawFunc = lambda *args, **kwargs: drawFrequencies_AdaptedRadius(*args, **kwargs, randn_mat_0_1=randn_mat_0_1, seed=seed, keep_splitted=keep_splitted, R_seeds=R_seeds)
    elif drawType.lower() in ["drawfrequencies_adapted_kmeans", "adaptedradius_kmeans", "adapted_radius_kmeans", "arkm",
                              "ar-km"]:
        drawFunc = lambda _a, _b, _c: drawFrequencies_AdaptedRadius(_a, _b, _c, KMeans=True, randn_mat_0_1=randn_mat_0_1, seed=seed, keep_splitted=keep_splitted, R_seeds=R_seeds)
    else:
        raise ValueError("drawType not recognized")

    # Handle no input
    if Sigma is None:
        Sigma = np.identity(d)
    else:
        Sigma = Sigma

    # Handle
    if isinstance(Sigma, np.ndarray) or is_number(Sigma):
        Omega = drawFunc(d, m, Sigma)

    # Handle mixture-type input
    elif isinstance(Sigma, tuple):
        assert keep_splitted is False, "Splitted output not implemented for mixture of Gaussians sampling"
        (w, cov) = Sigma  # unpack
        K = w.size
        # Assign the frequencies to the mixture components
        assignations = np.random.choice(K, m, p=w)
        Omega = np.zeros((d, m))
        for k in range(K):
            active_index = (assignations == k)
            if any(active_index):
                Omega[:, np.where(active_index)[0]] = drawFunc(d, active_index.sum(), cov[k])

    else:
        raise ValueError("Sigma not recognized")

    # If needed, overwrite the integer entries
    if nb_cat_per_dim is not None:
        assert keep_splitted is False, "Splitted output not implemented for mixture of Gaussians sampling"
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

    if return_torch:
        if keep_splitted:
            return torch.from_numpy(Omega[0]), torch.from_numpy(Omega[1]), torch.from_numpy(Omega[2])
        else:
            return torch.from_numpy(Omega)
    else:
        return Omega


def rebuild_Omega_from_sigma_direction_R(sig, dir, R, math_module=torch):
    """
    Return the reconstructed frequencies matrix obtained from:
        - a (sequence of) sigmas
        - a set of directions
        - a (set of) set of radii

    In the final matrix is constructed like:

    frequencies = []
    for R in set of radii vectors:
        for sigma in set of sigmas:
            frequencies.concat_to_the_end(frequencies(R, sigma))

    Parameters
    ----------
    sig:
        A sequence of sigmas. Shape: (S,)
    dir:
        A matrix of directions sampled on the unitsphere. Shape: (D, M)
    R:
        A set of radii. Shape: (M, R)
    math_module:
        backend module torch or numpy

    Returns
    -------
        The reconstructed matrix of shape (D, M*S*R)
    """

    dr = math_module.einsum("ij,jk->ikj", dir, R)
    r = math_module.einsum("l,ikj->iklj", sig, dr)
    return r.reshape((dir.shape[0], -1))
