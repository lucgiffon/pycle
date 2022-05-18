import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


# cleaning add function to display sigma estimation here
def plotGMM(X=None, P=None, dims=(0, 1), d=2, proportionInGMM=None):
    """
    Plots a Gaussian mixture model (and associated data) in 2 dimensions.

    Parameters
    ----------
    X: (n,d)-numpy array, the dataset of n examples in dimension d (optional)
    P: a a tuple (w,mus,Sigmas) of three numpy arrays describing the Gaussian mixture model, where
        - w:      (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    dims: tuple of two integers (default (0,1)), lists the 2 dimensions out of the d in X to plot


    Additional Parameters
    ---------------------
    d: ambient dimension (used to compute the size of the contour)
    proportionInGMM: proportion of data to include in the GMM ellipses (default 95%)
    """
    # todo To finish

    if P is not None:
        (w, mus, Sigmas) = P  # Unpack
        K = w.size

    (w, mus, Sigmas) = P  # Unpack
    K = w.size
    dim0, dim1 = dims
    if proportionInGMM is None:
        # for 95, d = 2%
        cst = 2 * np.sqrt(5.991)
    else:
        # check https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        cst = 2 * np.sqrt(chi2.isf(1 - proportionInGMM,
                                   d))
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, dim0], X[:, dim1], s=1, alpha=0.15)
    ax = plt.gca()

    for k in range(K):
        mu = mus[k]
        # Compute eigenvalues
        (lam, v) = np.linalg.eig(Sigmas[k][[dim0, dim1], :][:, [dim0, dim1]])
        # Sort
        v_max = v[:, np.argmax(lam)]

        plt.scatter(mu[dim0], mu[dim1], s=200 * w[k], c='r')

        wEll = cst * np.sqrt(lam.max())
        hEll = cst * np.sqrt(lam.min())

        with np.errstate(divide='ignore'):  # ignore divide by zero warning
            angle = np.arctan(v_max[1] / v_max[0]) * 180 / (np.pi)
        # if np.abs(v_max[0]) >= np.abs(v_max[1])*1e-9:
        #    angle = np.arctan(v_max[1]/v_max[0])*180/(np.pi)
        # else:
        #    angle = 0

        ellipse = Ellipse(xy=mu, width=wEll, height=hEll, angle=angle,
                          edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)

    plt.show()

    return
