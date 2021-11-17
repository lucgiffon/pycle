import numpy as np
import scipy.stats
import torch

from torch.autograd.function import Function
from lightonml.encoding.base import SeparatedBitPlanEncoder, SeparatedBitPlanDecoder

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


def enc_dec_fct(fct, x, precision_encoding=8):
    """
    Encode x in binary for OPU transformation then decode.
    This function just makes a transformation of x with an encoding/decoding (noisy) step.

    :param fct: fct that can take x as input
    :param x:
    :param precision_encoding:
    :return:
    """
    encoder = SeparatedBitPlanEncoder(precision=precision_encoding)
    x_enc = encoder.fit_transform(x)
    y_enc = fct(x_enc)
    decoder = SeparatedBitPlanDecoder(**encoder.get_params())
    y_dec = decoder.transform(y_enc)
    return y_dec


def only_quantification_fct(fct, x, precision_encoding=8):
    """
    Only make quantification of x before applying the fct.

    :param fct: fct that can take x as input
    :param x:
    :param precision_encoding:
    :return:
    """
    encoder = SeparatedBitPlanEncoder(precision=precision_encoding)
    x_enc = encoder.fit_transform(x)
    decoder = SeparatedBitPlanDecoder(**encoder.get_params())
    x_dec = decoder.transform(x_enc)
    y_dec = fct(x_dec)
    return y_dec


class LinearFunctionEncDec(Function):

    @staticmethod
    # def forward(ctx, input, weight):
    def forward(ctx, input, weight, quantif=False, enc_dec=False):
        assert not (quantif and enc_dec)

        ctx.save_for_backward(input, weight)
        if quantif or enc_dec:
            encoder = SeparatedBitPlanEncoder(precision=8)
            x_enc = encoder.fit_transform(input.data)
            decoder = SeparatedBitPlanDecoder(**encoder.get_params())
        else:
            x_enc = input

        if quantif and not enc_dec:
            # in case only quantification of input is requiered (testing purposes):
            # make quantification/dequantification directly
            x_enc = decoder.transform(x_enc)

        # if ever using the opu here: careful with the type of x_enc and y_dec

        y_dec = x_enc.to(weight.dtype).mm(weight)

        if not quantif and enc_dec:
            # standard scenario: dequantification happens after the transformation
            y_dec = decoder.transform(y_dec)

        return y_dec.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_output.mm(weight.t())
        # grad_weight = grad_output.t().mm(input)

        # first None is for the weights which have fixed values
        # two last None correspond to `quantif` and `enc_dec` arguments in forward pass
        return grad_input, None, None, None



class MultiSigmaARFrequencyMatrixLinApEncDec(Function):

    @staticmethod
    def forward(ctx, input, SigFacts, directions, R, quantif=False, enc_dec=False):
        assert not (quantif and enc_dec)

        ctx.save_for_backward(input, SigFacts, directions, R)
        if quantif or enc_dec:
            encoder = SeparatedBitPlanEncoder(precision=8)
            x_enc = encoder.fit_transform(input.data)
            decoder = SeparatedBitPlanDecoder(**encoder.get_params())
        else:
            x_enc = input

        if quantif and not enc_dec:
            # in case only quantification of input is requiered (testing purposes):
            # make quantification/dequantification directly
            x_enc = decoder.transform(x_enc)

        # if ever using the opu here: careful with the type of x_enc and y_dec

        # y_dec = x_enc.to(weight.dtype).mm(weight)
        weight_dtype = torch.promote_types(torch.promote_types(SigFacts.dtype, directions.dtype), R.dtype)
        y_dec = x_enc.to(weight_dtype).mm(directions) * R
        y_dec = torch.einsum("ij,jkl->kil", SigFacts.unsqueeze(-1), y_dec.unsqueeze(0)).reshape((input.shape[0], directions.shape[1] * SigFacts.shape[0]))

        if not quantif and enc_dec:
            # standard scenario: dequantification happens after the transformation
            y_dec = decoder.transform(y_dec)

        return y_dec.to(weight_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, SigFacts, directions, R = ctx.saved_tensors
        grad_output = (grad_output).reshape((grad_output.shape[0], SigFacts.shape[0], directions.shape[1]))
        grad_input = (torch.einsum("i,kil->kl", SigFacts, grad_output) * R).mm(directions.t())
        # grad_input = grad_output.mm(weight.t())

        # the three first Nones are for the weights which have fixed values
        # two last None correspond to `quantif` and `enc_dec` arguments in forward pass
        return grad_input, None, None, None, None, None