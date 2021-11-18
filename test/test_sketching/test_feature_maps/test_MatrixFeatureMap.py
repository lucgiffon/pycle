import pytest
import numpy as np
import torch

import pycle.sketching.frequency_sampling

from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.feature_maps.OPUFeatureMap import calibrate_lin_op, OPUFeatureMap
from pycle.utils import enc_dec_fct
from pycle.utils.datasets import generatedataset_GMM


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

@pytest.fixture
def X(my_dim):
    nb_clust = 5
    np.random.seed(20)  # easy
    # np.random.seed(722233)
    nb_sample = 2000  # Number of samples we want to generate
    # We use the generatedataset_GMM method from pycle (we ask that the entries are <= 1, and imbalanced clusters)
    X = generatedataset_GMM(my_dim, nb_clust, nb_sample, normalize='l_inf-unit-ball', balanced=False)
    X = torch.from_numpy(X).double()

    return X


def test_MatrixFeatureMap_multi_sigma(my_dim, X):
    for use_torch in [True]:
        for nb_repeats in [1, 3]:
            print(f"nb_repeats={nb_repeats}")
            print(f"use_torch={use_torch}")
            sampling_method = "ARKM"
            sketch_dim = my_dim * 2
            Sigma = np.array([0.876] * nb_repeats)
            nb_input = 3
            seed = 0

            lst_omega = [sifact, directions, R] = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                                                                     seed=seed, keep_splitted=True, return_torch=use_torch)
            # lst_omega = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
            #                                                            seed=seed, keep_splitted=False)

            lst_omega = tuple(lst_omega)

            MFM = MatrixFeatureMap(f="ComplexExponential", Omega=lst_omega, use_torch=use_torch)

            input_mat = np.random.randn(nb_input, my_dim)
            input_mat = torch.Tensor(input_mat)
            mfm_output = MFM(input_mat)
            assert mfm_output.shape[-1] == nb_repeats*sketch_dim
            if use_torch:
                assert (np.tile(mfm_output[..., :sketch_dim], nb_repeats) == mfm_output.numpy()).all()
            else:
                assert (np.tile(mfm_output[..., :sketch_dim], nb_repeats) == mfm_output).all()

            z = pycle.sketching.computeSketch(X, MFM)
            print(z.shape)