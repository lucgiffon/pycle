import numpy as np


def mu_estimation_ones(lin_op_fct, in_dim):
    """
    Estimate the mean coefficient of lin_op_fct by probing it with ones.
    :param lin_op_fct:
    :param in_dim:
    :return:
    """
    ones = np.ones(in_dim)
    y = lin_op_fct(ones)
    out_dim = y.shape[-1]
    return np.sum(y) / (out_dim * in_dim)


def var_estimation_ones(lin_op_fct, dim):
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with ones.
    :param lin_op_fct:
    :param dim:
    :return:
    """
    ones = np.ones(dim)
    y = lin_op_fct(ones)
    D_var = np.var(y)
    return D_var / dim


def var_estimation_randn(lin_op_fct, dim, n_iter=1, mu_not_zero=False):
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with norm 1 vectors.
    :param lin_op_fct:
    :param dim:
    :return:
    """
    x = np.random.randn(n_iter, dim)
    y = lin_op_fct(x)
    D_var_plus_mu = np.var(y)
    if mu_not_zero:
        mu = mu_estimation_ones(lin_op_fct, dim)
        var = D_var_plus_mu / dim - (mu ** 2)
    else:
        var = D_var_plus_mu / dim
    return var


def var_estimation_any(lin_op_fct, dim, n_iter=1):
    """
    Estimate the var of the coefficients in lin_op_fct by probing it with any vector of known norm.

    Only works if mu is zero.
    :param lin_op_fct:
    :param dim:
    :return:
    """
    # only works if mu is zero
    X = np.random.randn(n_iter, dim)
    X_norm_2 = np.linalg.norm(X, axis=1).reshape(n_iter, -1)  # the vector of all samples norms
    X /= X_norm_2  # samples in X are now of norm 1
    Y = lin_op_fct(X)  # linear transformation of X
    Y_squared = Y ** 2
    Y_norm_2 = np.sum(Y_squared)  # sum of all the Y_{i,j}^2
    var = Y_norm_2 / Y.size  # get the mean value of Y_{i,j}^2
    return var
