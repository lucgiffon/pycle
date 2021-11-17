import pytest
import numpy as np
import torch

import pycle.sketching.frequency_sampling

from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.feature_maps.OPUFeatureMap import calibrate_lin_op, OPUFeatureMap
from pycle.utils import enc_dec_fct


@pytest.fixture
def my_dim():
    dim = 5
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


def test_MatrixFeatureMap_multi_sigma(my_dim):
    sampling_method = "ARKM"
    sketch_dim = my_dim * 2
    Sigma = 0.876
    nb_input = 3
    seed = 0

    lst_omega = [sifact, directions, R] = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                               seed=seed, keep_splitted=True)

    lst_omega = list(lst_omega)
    nb_repeats = 4
    lst_omega[0] = np.array([lst_omega[0]] * nb_repeats)
    # lst_omega = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
    #                                                            seed=seed, keep_splitted=False)
    for i, elm in enumerate(lst_omega):
        lst_omega[i] = torch.tensor(lst_omega[i])

    lst_omega = tuple(lst_omega)

    MFM = MatrixFeatureMap(f="ComplexExponential", Omega=lst_omega, use_torch=True)

    input_mat = np.random.randn(nb_input, my_dim)
    input_mat = torch.Tensor(input_mat)
    mfm_output = MFM(input_mat)
    assert mfm_output.shape[-1] == nb_repeats*sketch_dim
    assert (np.tile(mfm_output[..., :sketch_dim], nb_repeats) == mfm_output.numpy()).all()