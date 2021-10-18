"""
Sketching functions
"""

import sys

import numpy as np
import torch

from pycle.sketching.feature_maps.FeatureMap import FeatureMap
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap


def computeSketch(dataset, featureMap, datasetWeights=None, batch_size=None, use_torch=False):
    """
    Computes the sketch of a dataset given a generic feature map.

    More precisely, evaluates
        z = sum_{x_i in X} w_i * Phi(x_i)
    where X is the dataset, Phi is the sketch feature map, w_i are weights assigned to the samples (typically 1/n).

    Arguments:
        - dataset        : (n,d) numpy array, the dataset X: n examples in dimension d
        - featureMap     : the feature map Phi, given as one of the following:
            -- a function, z_x_i = featureMap(x_i), where x_i and z_x_i are (n,)- and (m,)-numpy arrays, respectively
            -- a FeatureMap instance (e.g., constructed as featureMap = SimpleFeatureMap("complexExponential",Omega) )
        - datasetWeights : (n,) numpy array, optional weigths w_i in the sketch (default: None, corresponds to w_i = 1/n)

    Returns:
        - sketch : (m,) numpy array, the sketch as defined above
    """
    # TODOs:
    # - add possibility to specify classes and return one sketch per class
    # - defensive programming
    # - efficient implementation, take advantage of parallelism

    (n, d) = dataset.shape  # number of samples, dimension

    # Determine the sketch dimension and sanity check: the dataset is nonempty and the map works
    if isinstance(featureMap, FeatureMap):  # featureMap is the argument, FeatureMap is the class
        m = featureMap.m
    else:
        try:
            m = featureMap(dataset[0]).shape[0]
        except Exception:
            raise ValueError("Unexpected error while calling the sketch feature map:", sys.exc_info()[0])

    # Split the batches
    if batch_size is None:
        batch_size = int(1e6 / m)  # Rough heuristic, best batch size will vary on different machines
    nb_batches = int(np.ceil(n / batch_size))

    if use_torch:
        sketch = torch.zeros(m)
        sum_arg = {"dim": 0}
    else:
        sketch = np.zeros(m)
        sum_arg = {"axis": 0}
    if datasetWeights is None:
        for b in range(nb_batches):
            sketch = sketch + featureMap(dataset[b * batch_size:(b + 1) * batch_size]).sum(**sum_arg)
        sketch /= n
    else:
        sketch = datasetWeights @ featureMap(dataset)
    return sketch


def sensisitivty_sketch(featureMap, n=1, DPdef='UDP', sensitivity_type=1):
    """
    Computes the sensitity of a provided sketching function.

    The noisy sketch operator A(X) is given by
        A(X) := (1/n)*[sum_{x_i in X} featureMap(x_i)] + w
    where w is Laplacian or Gaussian noise.

    Arguments:
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a FeatureMap object with a known sensitivity (i.e., complex exponential or universal quantization periodic map)
            -- (m,featureMapName,c_normalization): tuple (deprectated, only useful for code not supporting FeatureMap objects),
                that should contain:
                -- m: int, the sketch dimension
                -- featureMapName: string, name of sketch feature function f, values supported:
                    -- 'complexExponential' (f(t) = exp(i*t))
                    -- 'universalQuantization_complex' (f(t) = sign(exp(i*t)))
                -- c_normalization: real, the constant before the sketch feature function (e.g., 1. (default), 1./sqrt(m),...)
        - n: int, number of sketch contributions being averaged (default = 1, useful to add noise on n independently)
        - DPdef: string, name of the Differential Privacy variant considered, i.e. the neighbouring relation ~:
            -- 'remove', 'add', 'remove/add', 'UDP' or 'standard': D~D' iff D' = D U {x'} (or vice versa)
            -- 'replace', 'BDP': D~D' iff D' = D \ {x} U {x'} (or vice versa)
        - sensitivity_type: int, 1 (default) for L1 sensitivity, 2 for L2 sensitivity.


    Returns: a positive real, the L1 or L2 sensitivity of the sketching operator defined above.

    Cfr: Differentially Private Compressive K-means, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682829.
    """

    # schellekensvTODO include real cases (cosine, real universal quantization)

    # The sensitivity is of the type: c_feat*c_
    if isinstance(featureMap, FeatureMap):
        m = featureMap.m
        featureMapName = featureMap.name
        c_normalization = featureMap.c_norm
    elif (isinstance(featureMap, tuple)) and (len(featureMap) == 3):
        (m, featureMapName, c_normalization) = featureMap
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')

    # Sensitivity is given by S = c_featureMap * c_sensitivity_type * c_DPdef, check all three conditions (ughh)
    if featureMapName.lower() == 'complexexponential':
        if sensitivity_type == 1:
            if DPdef.lower() in ['remove', 'add', 'remove/add', 'standard', 'udp']:
                return m * np.sqrt(2) * (c_normalization / n)
            elif DPdef.lower() in ['replace', 'bdp']:
                return 2 * m * np.sqrt(2) * (c_normalization / n)
        elif sensitivity_type == 2:
            if DPdef.lower() in ['remove', 'add', 'remove/add', 'standard', 'udp']:
                return np.sqrt(m) * (c_normalization / n)
            elif DPdef.lower() in ['replace', 'bdp']:
                return np.sqrt(m) * np.sqrt(2) * (c_normalization / n)
    elif featureMapName.lower() == 'universalquantization_complex':  # Assuming normalized in [-1,+1], schellekensvTODO check real/complex case?
        if sensitivity_type == 1:
            if DPdef.lower() in ['remove', 'add', 'remove/add', 'standard', 'udp']:
                return m * 2 * (c_normalization / n)
            elif DPdef.lower() in ['replace', 'bdp']:
                return 2 * m * 2 * (c_normalization / n)
        elif sensitivity_type == 2:
            if DPdef.lower() in ['remove', 'add', 'remove/add', 'standard', 'udp']:
                return np.sqrt(m) * np.sqrt(2) * (c_normalization / n)
            elif DPdef.lower() in ['replace', 'bdp']:
                return np.sqrt(2) * np.sqrt(m) * np.sqrt(2) * (c_normalization / n)
    print(sensitivity_type)
    raise Exception(
        'You provided ({},{});\nThe sensitivity for this (feature map,DP definition) combination is not implemented.'.format(
            featureMapName.lower(), DPdef.lower()))
    return None


def computeSketch_DP(dataset, featureMap, epsilon, delta=0, DPdef='UDP', useImproveGaussMechanism=True,
                     budget_split_num=None):
    """
    Computes the Differentially Private sketch of a dataset given a generic feature map.

    More precisely, evaluates the DP sketching mechanism:
        z = ( sum_{x_i in X} Phi(x_i) + w_num )/( |X| + w_den )
    where X is the dataset, Phi is the sketch feature map, w_num and w_den are Laplacian or Gaussian random noise.

    Arguments:
        - dataset        : (n,d) numpy array, the dataset X: n examples in dimension d
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a FeatureMap object with a known sensitivity (i.e., complex exponential or universal quantization periodic map)
            -- (featureMap(x_i),m,featureMapName,c_normalization): tuple (deprectated, only useful for old code),
                that should contain:
                -- featMap: a function, z_x_i = featMap(x_i), where x_i and z_x_i are (n,)- and (m,)-numpy arrays, respectively
                -- m: int, the sketch dimension
                -- featureMapName: string, name of sketch feature function f, values supported:
                    -- 'complexExponential' (f(t) = exp(i*t))
                    -- 'universalQuantization' (f(t) = sign(exp(i*t)))
                -- c_normalization: real, the constant before the sketch feature function (e.g., 1. (default), 1./sqrt(m),...)
        - epsilon: real > 0, the privacy parameter epsilon
        - delta:  real >= 0, the privacy parameter delta in approximate DP; if delta=0 (default), we have "pure" DP.
        - DPdef: string, name of the Differential Privacy variant considered, i.e. the neighbouring relation ~:
            -- 'remove', 'add', 'remove/add', 'UDP' or 'standard' (default): D~D' iff D' = D U {x'} (or vice versa)
            -- 'replace', 'BDP': D~D' iff D' = D \ {x} U {x'} (or vice versa)
        - useImproveGaussMechanism: bool, if True (default) use the improved Gaussian mechanism[1] rather than usual bounds[2].
        - budget_split_num: 0 < real < 1, fraction of epsilon budget to allocate to the numerator (ignored in BDP).
                            By default, we assign a fraction of (2*m)/(2*m+1) on the numerator.

    Returns:
        - sketch : (m,) numpy array, the differentially private sketch as defined above
    """

    # Extract dataset size
    (n, d) = dataset.shape

    # Compute the nonprivate, usual sketch
    if isinstance(featureMap, FeatureMap):
        z_clean = computeSketch(dataset, featureMap)
    elif (isinstance(featureMap, tuple)) and (callable(featureMap[0])):
        featMap = featureMap[0]
        featureMap = featureMap[1:]
        z_clean = computeSketch(dataset, featMap)

    if epsilon == np.inf:  # Non-private
        return z_clean

    useBDP = DPdef.lower() in ['replace', 'bdp']  # otherwise assume UDP, schellekensvTODO DEFENSIVE

    # We will need the sketch size
    m = z_clean.size

    # Split privacy budget
    if useBDP:  # Then no noise on the denom
        budget_split_num = 1.
    elif budget_split_num is None:
        budget_split_num = (2 * m) / (2 * m + 1)
    # schellekensvTODO defensive programming to block budget split > 1?
    epsilon_num = budget_split_num * epsilon

    # Compute numerator noise
    if delta > 0:
        # Gaussian mechanism
        S = sensisitivty_sketch(featureMap, DPdef=DPdef, sensitivity_type=2)  # L2

        if useImproveGaussMechanism:  # Use the sharpened bounds
            from pycle.third_party import calibrateAnalyticGaussianMechanism
            sigma = calibrateAnalyticGaussianMechanism(epsilon_num, delta, S)
        else:  # use usual bounds
            if epsilon >= 1: raise Exception(
                'WARNING: with epsilon >= 1 the sigma bound doesn\'t hold! Privacy is NOT ensured!')
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * S / epsilon_num
        noise_num = np.random.normal(scale=sigma, size=m) + 1j * np.random.normal(scale=sigma,
                                                                                  size=m)  # schellekensvTODO real
    else:
        # Laplacian mechanism
        S = sensisitivty_sketch(featureMap, DPdef=DPdef, sensitivity_type=1)  # L1
        beta = S / epsilon_num  # L1 sensitivity/espilon
        noise_num = np.random.laplace(scale=beta, size=m) + 1j * np.random.laplace(scale=beta, size=m)

        # Add denominator noise if needed
    if useBDP:  # Then no noise on the denom
        return z_clean + (noise_num / n)
    else:
        num = (z_clean * n) + noise_num
        beta_den = 1 / (epsilon - epsilon_num)  # rest of the privacy budget
        den = n + np.random.laplace(scale=beta_den)
        return num / den


def fourierSketchOfGaussian(mu, Sigma, Omega, xi=None, scst=None):
    res = np.exp(1j * (mu @ Omega) - np.einsum('ij,ij->i', np.dot(Omega.T, Sigma), Omega.T) / 2.)
    if xi is not None:
        res = res * np.exp(1j * xi)
    if scst is not None:  # Sketch constant, eg 1/sqrt(m)
        res = scst * res
    return res


def fourierSketchOfGMM(GMM, featureMap):
    """Returns the complex exponential sketch of a Gaussian Mixture Model

    Parameters
    ----------
    GMM: (weigths,means,covariances) tuple, the Gaussian Mixture Model, with
        - weigths:     (K,)-numpy array containing the weigthing factors of the Gaussians
        - means:       (K,d)-numpy array containing the means of the Gaussians
        - covariances: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    featureMap: the sketch the sketch featureMap (Phi), provided as either:
        - a SimpleFeatureMap object (i.e., complex exponential or universal quantization periodic map)
        - (Omega,xi): tuple with the (d,m) Fourier projection matrix and the (m,) dither (see above)

    Returns
    -------
    z: (m,)-numpy array containing the sketch of the provided GMM
    """
    # Parse GMM input
    (w, mus, Sigmas) = GMM
    K = w.size

    # Parse featureMap input
    if isinstance(featureMap, MatrixFeatureMap):
        Omega = featureMap.Omega
        xi = featureMap.xi
        d = featureMap.d
        m = featureMap.m
        scst = featureMap.c_norm  # Sketch normalization constant, e.g. 1/sqrt(m)
    elif isinstance(featureMap, tuple):
        (Omega, xi) = featureMap
        (d, m) = Omega.shape
        scst = 1.  # This type of argument passing does't support different normalizations
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')

    z = 1j * np.zeros(m)
    for k in range(K):
        z += fourierSketchOfGaussian(mus[k], Sigmas[k], Omega, xi, scst)
    return z


def fourierSketchOfBox(box, featureMap, nb_cat_per_dim=None, dimensions_to_consider=None):
    """Returns the complex exponential sketch of the indicator function on a parallellipiped (box).
    For dimensions that flagged as integer, considers the indicator on a set of integers instead.

    Parameters
    ----------
    box: (d,2)-numpy array, the boundaries of the box (x in R^d is in the box iff box[i,0] <= x_i <= box[i,1])
    featureMap: the sketch the sketch featureMap (Phi), provided as either:
        - a SimpleFeatureMap object (is assumed to use the complex exponential map)
        - (Omega,xi): tuple with the (d,m) Fourier projection matrix and the (m,) dither (see above)

    Additional Parameters
    ---------------------
    nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous.
    dimensions_to_consider: array of ints (between 0 and d-1), [0,1,...d-1] by default.
                    The box is restricted to the prescribed dimensions.
                    This is helpful to solve problems on a subsets of all dimensions.


    Returns
    -------
    z: (m,)-numpy array containing the sketch of the indicator function on the provided box
    """
    ## Parse input
    # Parse box input
    (d, _) = box.shape
    c_box = (box[:, 1] + box[:, 0]) / 2  # Center of the box in each dimension
    l_box = (box[:, 1] - box[:, 0]) / 2  # Length (well, half of the length) of the box in each dimension

    # Parse featureMap input
    if isinstance(featureMap, MatrixFeatureMap):
        Omega = featureMap.Omega
        xi = featureMap.xi
        d = featureMap.d
        m = featureMap.m
        scst = featureMap.c_norm  # Sketch normalization constant, e.g. 1/sqrt(m)
    elif isinstance(featureMap, tuple):
        (Omega, xi) = featureMap
        (d, m) = Omega.shape
        scst = 1.  # This type of argument passing does't support different normalizations
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')

    # Parse nb_cat_per_dim
    if nb_cat_per_dim is None:
        nb_cat_per_dim = np.zeros(d)

    # Parse dimensions to consider
    if dimensions_to_consider is None:
        dimensions_to_consider = np.arange(d)

    ## Compute sketch
    z = scst * np.exp(1j * xi)
    for i in dimensions_to_consider:
        mask_valid = np.abs(Omega[i]) > 1e-15
        # CHECK IF INTEGER OR CONTINUOUS
        if nb_cat_per_dim[i] > 0:
            low_int = box[i, 0]
            high_int = box[i, 1] + 1  # python counting convention
            C = high_int - low_int
            newTerm = np.ones(m) + 1j * np.zeros(m)
            newTerm[mask_valid] = (1 / C) * (
                        np.exp(1j * Omega[i, mask_valid] * high_int) - np.exp(1j * Omega[i, mask_valid] * low_int)) / (
                                              np.exp(1j * Omega[i, mask_valid]) - 1)
            z *= newTerm
        else:
            # If continuous
            # To avoid divide by zero error, use that lim x->0 sin(a*x)/x = a
            sincTerm = np.zeros(m)

            sincTerm[mask_valid] = np.sin(Omega[i, mask_valid] * l_box[i]) / Omega[i, mask_valid]
            sincTerm[~mask_valid] = l_box[i]

            z *= 2 * np.exp(1j * Omega[i] * c_box[i]) * sincTerm
    return z

### TODOS FOR SKETCHING.PY

# Short-term:
#  - Add support of private sketching for the real variants of the considered maps
#  - Add the square nonlinearity, for sketching for PCA for example

# Long-term:
# - Fast sketch computation