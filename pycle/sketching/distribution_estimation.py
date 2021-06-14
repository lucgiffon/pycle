import numpy as np


def mu_estimation_ones(lin_op_fct, in_dim):
    ones = np.ones(in_dim)
    y = lin_op_fct(ones)
    out_dim = y.shape[-1]
    return np.sum(y) / (out_dim * in_dim)


def var_estimation_ones(lin_op_fct, dim):
    ones = np.ones(dim)
    y = lin_op_fct(ones)
    D_var = np.var(y)
    return D_var / dim


def var_estimation_randn(lin_op_fct, dim, n_iter=1, mu_not_zero=False):
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
    # only works if mu is zero
    X = np.random.rand(n_iter, dim)
    X_norm_2 = np.linalg.norm(X, axis=1).reshape(n_iter, -1)
    X /= X_norm_2
    Y = lin_op_fct(X)
    Y_norm_2 = np.linalg.norm(Y) ** 2
    var = Y_norm_2 / Y.size
    return var


def main():
    pass


if __name__ == "__main__":
    main()
