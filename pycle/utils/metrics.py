"""
This module contains evaluation metrics.
"""

import numpy as np
import scipy.spatial
import scipy.stats


def SSE(X,C):
    """Computes the Sum of Squared Errors of some centroids on a dataset, given by
        SSE(X,C) = sum_{x_i in X} min_{c_k in C} ||x_i-c_k||_2^2.

    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - C: (K,d)-numpy array, the K centroids in dimension d

    Returns:
        - SSE: real, the SSE score defined above
    """
    distances = scipy.spatial.distance.cdist(X, C, 'sqeuclidean')

    return np.min(distances,axis=1).sum()


def indicator_vector(X,C):
    """Computes the indicator vector giving labels of the closest c in C to every x in X

    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - C: (K,d)-numpy array, the K centroids in dimension d

    Returns:
        - indicator vector: (n)-array of the indices in C
    """
    distances = scipy.spatial.distance.cdist(X, C, 'sqeuclidean')

    return np.argmin(distances, axis=1)


def loglikelihood_GMM(P,X,robust = True):
    """Computes the loglikelihood of GMM model P on data X, defined as follows:
        loglikelihood = (1/n) * sum_{i=1..n} log(sum_{k=1..K} (w_k)*N(x_i ; mu_k, Sigma_k) )

    Arguments:
        - P: tuple of three numpy arrays describing the GMM model of form (w,mus,Sigmas)
            - w      : (K,)-numpy array, the weights of the K Gaussians (should sum to 1)
            - mus    : (K,d)-numpy array containing the means of the Gaussians
            - Sigmas : (K,d,d)-numpy array containing the covariance matrices of the Gaussians
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - robust: bool (default = True), if True, avoids -inf output due to very small probabilities
                  (note: execution will be slower)

    Returns:
        - loglikelihood: real, the loglikelihood value defined above
    """

    # Unpack
    (w,mu,Sig) = P
    (K,d) = mu.shape

    logp = np.zeros(X.shape[0])
    p = np.zeros(X.shape[0])

    try:
        for k in range(K):
            p += w[k]*scipy.stats.multivariate_normal.pdf(X, mean=mu[k], cov=Sig[k], allow_singular=False)
        with np.errstate(divide='ignore'): # ignore divide by zero warning
            logp = np.log(p)
    except np.linalg.LinAlgError:


        if robust:
            b = np.zeros(K)
            a = np.zeros(K)
            Sig_inv = np.zeros(Sig.shape)
            for k in range(K):
                a[k] = w[k]*((2*np.pi)**(-d/2))*(np.linalg.det(Sig[k])**(-1/2))
                Sig_inv[k] = np.linalg.inv(Sig[k])
            for i in range(p.size): # Replace the inf values due to rounding p to 0
                for k in range(K):
                    b[k] = -(X[i]-mu[k])@Sig_inv[k]@(X[i]-mu[k])/2
                lc = b.max()
                ebc = np.exp(b-lc)
                logp[i] = np.log(ebc@a) + lc
        else:
            raise np.linalg.LinAlgError('singular matrix')


    return np.mean(logp)


def symmKLdivergence_GMM(P1,P2,Neval = 500000):
    """Computes the symmetric KL divergence between two GMM densities."""
    # schellekensvTODO : a version that adapts Neval s.t. convergence?
    # Unpack
    (w1,mu1,Sig1) = P1
    (w2,mu2,Sig2) = P2
    K1 = w1.size
    K2 = w2.size

    Neval # Number of samples to evaluate the KL divergence

    # dumb implem for now, schellekensvTODO FAST IMPLEM!
    KLestimate = 0.

    assignations1 = np.random.choice(K1,Neval,p=w1)

    for k1 in range(K1):
        N1 = np.count_nonzero(assignations1 == k1)
        Y = np.random.multivariate_normal(mu1[k1], Sig1[k1], N1)

        P1 = np.zeros(N1)
        for k in range(K1):
            P1 += w1[k]*scipy.stats.multivariate_normal.pdf(Y, mean=mu1[k], cov=Sig1[k], allow_singular=True)
        P2 = np.zeros(N1)
        for k in range(K2):
            P2 += w2[k]*scipy.stats.multivariate_normal.pdf(Y, mean=mu2[k], cov=Sig2[k], allow_singular=True)

        # Avoid numerical instabilities
        P2[np.where(P2<=1e-25)[0]] = 1e-25
        KLestimate += np.sum(np.log(P1/P2))

    assignations2 = np.random.choice(K2,Neval,p=w2)

    for k2 in range(K2):
        N2 = np.count_nonzero(assignations2 == k2)
        Y = np.random.multivariate_normal(mu2[k2], Sig2[k2], N2)

        P1 = np.zeros(N2)
        for k in range(K1):
            P1 += w1[k]*scipy.stats.multivariate_normal.pdf(Y, mean=mu1[k], cov=Sig1[k], allow_singular=True)
        P2 = np.zeros(N2)
        for k in range(K2):
            P2 += w2[k]*scipy.stats.multivariate_normal.pdf(Y, mean=mu2[k], cov=Sig2[k], allow_singular=True)

        # Avoid numerical instabilities
        P1[np.where(P1<=1e-25)[0]] = 1e-25
        KLestimate += np.sum(np.log(P2/P1))

    KLestimate /= 2*Neval


    return KLestimate