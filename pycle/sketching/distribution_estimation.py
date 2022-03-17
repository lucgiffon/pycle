from typing import Callable

# cleaning check that these functions are compatible with the distribution estimator? and torchify if necessary.
import torch


def mu_estimation_ones(lin_op_fct: Callable, in_dim: int) -> torch.float:
    """
    Estimate the mean coefficient of lin_op_fct by probing it with ones.

    Parameters
    ----------
    lin_op_fct:
        Linear operator function taking torch.Tensor as input for which to estimate the mean coefficient.
    in_dim:
        Input dimension of the Linear operator function.

    Returns
    -------
        The estimated mean.
    """
    ones = torch.ones(in_dim)
    y = lin_op_fct(ones)
    out_dim = y.shape[-1]
    return torch.sum(y) / (out_dim * in_dim)


def var_estimation_ones(lin_op_fct: Callable, dim: int) -> torch.float:
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with ones.

    Parameters
    ----------
    lin_op_fct:
        Linear operator function taking torch.Tensor as input for which to estimate the mean coefficient.
    dim:
        Input dimension of the Linear operator function.

    Returns
    -------
        The estimated variance.
    """
    ones = torch.ones(dim)
    y = lin_op_fct(ones)
    D_var = torch.var(y)
    return D_var / dim


def var_estimation_randn(lin_op_fct: Callable, dim: int, n_iter: int = 1, mu_not_zero: bool = False) -> torch.float:
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with random N(0,1) vectors.


    Parameters
    ----------
    lin_op_fct:
        Linear operator function taking torch.Tensor as input for which to estimate the mean coefficient.
    dim:
        Input dimension of the Linear operator function.
    n_iter:
        Increase number of iteration for increased precision.
    mu_not_zero:
        If True, consider the mean coefficient not being equal to zero.

    Returns
    -------
        The estimated variance.
    """
    x = torch.randn(n_iter, dim)
    y = lin_op_fct(x)
    D_var_plus_mu = torch.var(y)
    if mu_not_zero:
        mu = mu_estimation_ones(lin_op_fct, dim)
        var = D_var_plus_mu / dim - (mu ** 2)
    else:
        var = D_var_plus_mu / dim
    return var


def var_estimation_any(lin_op_fct: Callable, dim: int, n_iter: int = 1) -> torch.float:
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with any vector of known norm.

    Only works if mu is zero.

    Parameters
    ----------
    lin_op_fct:
        Linear operator function taking torch.Tensor as input for which to estimate the mean coefficient.
    dim:
        Input dimension of the Linear operator function.
    n_iter:
        Increase number of iteration for increased precision.

    Returns
    -------
        The estimated variance.
    """
    # only works if mu is zero
    X = torch.randn(n_iter, dim)
    X_norm_2 = torch.norm(X, dim=1).reshape(n_iter, -1)  # the vector of all samples norms
    X /= X_norm_2  # samples in X are now of norm 1
    Y = lin_op_fct(X)  # linear transformation of X
    Y_squared = Y ** 2
    Y_norm_2 = torch.sum(Y_squared)  # sum of all the Y_{i,j}^2
    var = Y_norm_2 / torch.numel(Y)  # get the mean value of Y_{i,j}^2
    return var
