import pytest
import numpy as np
import torch

import pycle.sketching.frequency_sampling

from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice

from pycle.sketching import get_sketch_Omega_xi_from_aggregation
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.feature_maps.OPUFeatureMap import calibrate_lin_op, OPUFeatureMap
from pycle.utils.encoding_decoding import enc_dec_fct
from pycle.utils.datasets import generatedataset_GMM


@pytest.fixture
def my_dim():
    dim = 10
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

def test_split_unsplit(my_dim, X):
    sampling_method = "ARKM"
    sketch_dim = my_dim * 2
    Sigma = np.array([0.187, 0.36743])
    nb_input = 4
    seed = 0
    # r_seeds = [0]
    r_seeds = [0, 1]

    def build_mfm(splitted):
        Omega = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim,
                                                                   sketch_dim, Sigma,
                                                                   seed=seed,
                                                                   keep_splitted=splitted,
                                                                   return_torch=True,
                                                                   R_seeds=r_seeds)

        MFM = MatrixFeatureMap(f="complexexponential", Omega=Omega)
        return MFM

    input_mat = np.random.randn(nb_input, my_dim)
    input_mat = torch.Tensor(input_mat)

    MFM = build_mfm(splitted=False)
    mfm_output1 = MFM(input_mat)
    z1 = pycle.sketching.computeSketch(X, MFM)

    MFM2 = build_mfm(splitted=True)
    mfm_output2 = MFM2(input_mat)
    z2 = pycle.sketching.computeSketch(X, MFM2)

    assert torch.isclose(z1, z2).all()
    assert torch.isclose(mfm_output1, mfm_output2).all()

    MFM2.unsplit()
    mfm_output3 = MFM2(input_mat)
    z3 = pycle.sketching.computeSketch(X, MFM2)

    assert np.isclose(z1, z3).all()
    assert np.isclose(mfm_output1, mfm_output3).all()

    assert torch.isclose(MFM2.Omega, MFM.Omega).all()


def test_MatrixFeatureMap_retrieve_sketch_and_all(my_dim, X):
    sampling_method = "ARKM"
    sketch_dim = my_dim * 2
    nb_sigmas = 3
    base_Sigma = np.array([np.abs(np.random.randn(1)) for _ in range(nb_sigmas)]).flatten()
    seed = 0
    r_seeds = [0, 1, 2, 3, 4, 5]

    xi = torch.rand(sketch_dim*nb_sigmas*len(r_seeds))
    xi *= np.pi * 2
    # base omega
    sig, directions, R = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim,
                                                                                             sketch_dim, base_Sigma,
                                                                                             seed=seed,
                                                                                             keep_splitted=True,
                                                                                             return_torch=True,
                                                                                             R_seeds=r_seeds)
    lst_omega = (sig, directions, R)
    MFM = MatrixFeatureMap(f="complexexponential", Omega=lst_omega, xi=xi)
    # input_mat = np.random.randn(nb_input, my_dim)
    # input_mat = torch.Tensor(input_mat)
    # mfm_output = MFM(input_mat)
    aggregated_sketch = pycle.sketching.computeSketch(X, MFM)  # sketch 1

    # sketch with only one sigma and R
    indice_sigma = 2
    assert nb_sigmas > indice_sigma
    indice_R = 3
    assert indice_R < len(r_seeds)
    my_sigma = sig[indice_sigma]
    my_R_seed = r_seeds[indice_R]
    my_R = R[:, indice_R]

    begin_R = sketch_dim*nb_sigmas*indice_R
    end_R = sketch_dim*nb_sigmas*(indice_R+1)
    part_of_aggregated_sketch = aggregated_sketch[begin_R:end_R]
    my_xi = xi[begin_R:end_R]
    begin_sig = sketch_dim*indice_sigma
    end_sig = sketch_dim*(indice_sigma+1)
    part_of_aggregated_sketch = part_of_aggregated_sketch[begin_sig:end_sig]  # sketch 3
    my_xi = my_xi[begin_sig:end_sig]

    MFM = MatrixFeatureMap(f="complexexponential", Omega=(my_sigma, directions, my_R), xi=my_xi)
    localized_sketch = pycle.sketching.computeSketch(X, MFM)  # sketch 2

    assert localized_sketch.size() == part_of_aggregated_sketch.size()
    equality = torch.isclose(localized_sketch, part_of_aggregated_sketch)
    assert equality.all()

    part_of_aggregated_sketch_fct, (needed_sigma_fct, directions_fct, needed_R_fct), aggregated_xi_fct = \
        get_sketch_Omega_xi_from_aggregation(aggregated_sketch=aggregated_sketch.numpy(),
                                             aggregated_sigmas=sig.numpy(),
                                             directions=directions.numpy(),
                                             aggregated_R=R.numpy(),
                                             aggregated_xi=xi.numpy(),
                                             Omega=None,
                                             R_seeds=r_seeds,
                                             needed_sigma=my_sigma,
                                             needed_seed=my_R_seed,
                                             keep_all_sigmas=False,
                                             use_torch=True)  # sketch 3 bis

    assert localized_sketch.size() == part_of_aggregated_sketch_fct.size()
    equality = torch.isclose(localized_sketch, part_of_aggregated_sketch_fct)
    assert equality.all()

    assert torch.isclose(directions_fct, directions).all()
    assert torch.isclose(needed_sigma_fct, torch.Tensor([my_sigma])).all()
    assert torch.isclose(needed_R_fct, my_R).all()

    MFM = MatrixFeatureMap(f="complexexponential", Omega=(needed_sigma_fct, directions_fct, needed_R_fct), xi=aggregated_xi_fct)
    localized_sketch = pycle.sketching.computeSketch(X, MFM)  # sketch 2

    assert torch.isclose(part_of_aggregated_sketch_fct, localized_sketch).all()

    part_of_aggregated_sketch_fct, (needed_sigma_fct, directions_fct, needed_R_fct), aggregated_xi_fct = \
        get_sketch_Omega_xi_from_aggregation(aggregated_sketch=aggregated_sketch.numpy(),
                                             aggregated_sigmas=sig.numpy(),
                                             directions=directions.numpy(),
                                             aggregated_R=R.numpy(),
                                             aggregated_xi=xi.numpy(),
                                             Omega=None,
                                             R_seeds=r_seeds,
                                             needed_sigma=sig.numpy(),
                                             needed_seed=my_R_seed,
                                             keep_all_sigmas=True,
                                             use_torch=True)

    begin_R = sketch_dim * nb_sigmas * indice_R
    end_R = sketch_dim * nb_sigmas * (indice_R + 1)
    part_of_aggregated_sketch = aggregated_sketch[begin_R:end_R]
    assert torch.isclose(part_of_aggregated_sketch_fct, part_of_aggregated_sketch).all()


def test_MatrixFeatureMap_multi_sigma_multi_R(my_dim, X):
    for nb_sigmas in [0, 2, 1]:
        for nb_replicates in [3, 1]:
            print(f"nb_sigmas={nb_sigmas}")
            sampling_method = "ARKM"
            sketch_dim = my_dim * 2
            Sigma = 0.876
            if nb_sigmas != 0:
                Sigma = np.array([np.abs(np.random.randn(1)) for _ in range(nb_sigmas)]).flatten()
            else:
                nb_sigmas = 1
            nb_input = 4
            seed = 0
            r_seeds = [seed] * nb_replicates

            lst_omega = [sifact, directions, R] = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                                                                     seed=seed, keep_splitted=True, R_seeds=r_seeds, return_torch=True)
            # if nb_replicates != 1:
            #     if use_torch:
            #         R = R.repeat(nb_replicates, 1).T

            # lst_omega = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
            #                                                            seed=seed, keep_splitted=False)

            lst_omega = list(lst_omega)

            lst_omega[-1] = R

            lst_omega = tuple(lst_omega)

            MFM = MatrixFeatureMap(f="complexexponential", Omega=lst_omega)

            input_mat = np.random.randn(nb_input, my_dim)
            input_mat = torch.Tensor(input_mat)
            mfm_output = MFM(input_mat)
            assert mfm_output.shape[-1] == sketch_dim*nb_sigmas*nb_replicates
            if nb_sigmas == 1:
                assert (np.tile(mfm_output[..., :sketch_dim], nb_sigmas*nb_replicates) == mfm_output.numpy()).all()
            assert (np.tile(mfm_output[..., :sketch_dim*nb_sigmas], nb_replicates) == mfm_output.numpy()).all()

            z = pycle.sketching.computeSketch(X, MFM)
            print(z.shape)