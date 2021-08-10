import numpy as np
import pytest

from pycle.sketching.distribution_estimation import mu_estimation_ones, var_estimation_ones, var_estimation_randn, \
    var_estimation_any


@pytest.fixture
def my_dim():
    dim = 1000
    return dim


@pytest.fixture
def my_lst_lin_op(my_dim):
    lst_lin_op = [
        np.random.randn(my_dim, my_dim * 2),
        np.random.randn(my_dim, my_dim),
    ]
    return lst_lin_op


def test_mu_estimation_ones(my_lst_lin_op, my_dim):
    for idx, lin_op in enumerate(my_lst_lin_op):
        mean_oracle = np.mean(lin_op)
        mean_estimated = mu_estimation_ones(lambda x: x @ lin_op, in_dim=my_dim)
        assert np.isclose(mean_oracle, mean_estimated), f"{idx} lin_op failed"


def test_var_estimation_ones(my_lst_lin_op, my_dim):
    for idx, lin_op in enumerate(my_lst_lin_op):
        var_oracle = np.var(lin_op)
        var_estimated = var_estimation_ones(lambda x: x @ lin_op, dim=my_dim)
        assert np.isclose(var_oracle, var_estimated, atol=1), f"lin_op indice {idx} failed"


def test_var_estimation_randn(my_lst_lin_op, my_dim):
    for idx, lin_op in enumerate(my_lst_lin_op):
        var_oracle = np.var(lin_op)
        var_estimated = var_estimation_randn(lambda x: x @ lin_op, dim=my_dim, n_iter=1)
        assert np.isclose(var_oracle, var_estimated, atol=1), f"lin_op indice {idx} failed with n_iter=1"
        var_estimated = var_estimation_randn(lambda x: x @ lin_op, dim=my_dim, n_iter=10000)
        assert np.isclose(var_oracle, var_estimated, atol=1e-3), f"lin_op indice {idx} failed with n_iter=1000"


def test_var_estimation_any(my_lst_lin_op, my_dim):
    for idx, lin_op in enumerate(my_lst_lin_op):
        var_oracle = np.var(lin_op)
        var_estimated = var_estimation_any(lambda x: x @ lin_op, dim=my_dim, n_iter=1)
        assert np.isclose(var_oracle, var_estimated, atol=1), f"lin_op indice {idx} failed with n_iter=1"
        var_estimated = var_estimation_any(lambda x: x @ lin_op, dim=my_dim, n_iter=10000)
        assert np.isclose(var_oracle, var_estimated, atol=1e-3), f"lin_op indice {idx} failed with n_iter=1000"
