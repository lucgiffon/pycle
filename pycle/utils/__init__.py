import numpy as np
import scipy.stats

from pycle.utils.metrics import loglikelihood_GMM


def EM_GMM(X, K, max_iter=20, nRepetitions=1):
    """Usual Expectation-Maximization (EM) algorithm for fitting mixture of Gaussian models (GMM).

    Parameters
    ----------
    X: (n,d)-numpy array, the dataset of n examples in dimension d
    K: int, the number of Gaussian modes

    Additional Parameters
    ---------------------
    max_iter: int (default 20), the number of EM iterations to perform
    nRepetitions: int (default 1), number of independent EM runs to perform (returns the best)

    Returns: a tuple (w,mus,Sigmas) of three numpy arrays
        - w:      (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    # schellekensvTODO to improve:
    # - detect early convergence

    # Parse input
    (n,d) = X.shape
    lowb = np.amin(X,axis=0)
    uppb = np.amax(X,axis=0)


    bestGMM = None
    bestScore = -np.inf


    for rep in range(nRepetitions):
        # Initializations
        w = np.ones(K)
        mus = np.empty((K,d))
        Sigmas = np.empty((K,d,d)) # Covariances are initialized as random diagonal covariances, with folded Gaussian values
        for k in range(K):
            mus[k] = np.random.uniform(lowb,uppb)
            Sigmas[k] = np.diag(np.abs(np.random.randn(d)))
        r = np.empty((n,K)) # Matrix of posterior probabilities, here memory allocation only

        # Main loop
        for i in range(max_iter):
            # E step
            for k in range(K):
                r[:,k] = w[k]*scipy.stats.multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
            r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(

            # M step: 1) update w
            w = np.sum(r,axis=0)/n

            # M step: 2) update centers
            for k in range(K):
                mus[k] = r[:,k]@X/np.sum(r[:,k])

            # M step: 3) update Sigmas
            for k in range(K):
                # Dumb implementation
                num = np.zeros((d,d))
                for i in range(n):
                    num += r[i,k]*np.outer(X[i]-mus[k],X[i]-mus[k])
                Sigmas[k] = num/np.sum(r[:,k])

            # (end of one EM iteration)
        # (end of one EM run)
        newGMM = (w,mus,Sigmas)
        newScore = loglikelihood_GMM(newGMM, X)
        if newScore > bestScore:
            bestGMM = newGMM
            bestScore = newScore
    return bestGMM


class SingletonMeta(type):
    """
    Implements Singleton design pattern.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        When cls is "called" (asked to be initialised), return a new object (instance of cls) only if no object of that cls
        have already been created. Else, return the already existing instance of cls.

        args and kwargs parameters only affect the first call to constructor.

        :param args:
        :param kwargs:
        :return:
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]