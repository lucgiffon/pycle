import pytest
import numpy as np
from pycle.sketching.feature_maps.OPUFeatureMap import calibrate_lin_op


@pytest.fixture
def my_dim():
    dim = 100
    return dim


@pytest.fixture
def my_pow2dim():
    dim = 128
    return dim


@pytest.fixture
def my_lst_lin_op(my_dim, my_pow2dim):
    fac = 10
    lst_lin_op = [
        fac * np.random.randn(my_pow2dim, my_pow2dim * 2),
        fac * np.random.randn(my_pow2dim, my_pow2dim),
        fac * np.random.randn(my_dim, my_dim * 2),
        fac * np.random.randn(my_dim, my_dim),
    ]
    return lst_lin_op


def test_calibrate_lin_op(my_lst_lin_op):
    for idx, lin_op in enumerate(my_lst_lin_op):
        dim = lin_op.shape[0]
        calibrated_lin_op = calibrate_lin_op(lambda x: x @ lin_op, dim)
        assert np.isclose(calibrated_lin_op, lin_op).all(), f"idx {idx} failed"
