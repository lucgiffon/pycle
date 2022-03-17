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


def test_calibrate_lin_op(my_lst_lin_op):
    print()
    for nb_iter in [1, 3]:
        for idx, lin_op in enumerate(my_lst_lin_op):
            for use_torch in [False]:  # I don't really need it to be compatible with torch yet.
                print(f"use_torch: {use_torch}")
                if use_torch:
                    lin_op = torch.from_numpy(lin_op)
                dim = lin_op.shape[0]
                calibrated_lin_op = calibrate_lin_op(lambda x: x @ lin_op, dim, nb_iter=nb_iter)
                assert np.isclose(calibrated_lin_op, lin_op).all(), f"idx {idx} failed"
                var_lin_op = np.var(lin_op)
                var_calibrated = np.var(calibrated_lin_op)
                assert np.isclose(var_calibrated, var_lin_op).all(), f"idx {idx} failed var estimation"


def test_enc_dec_opu_transform():
    dim = 16
    opu = OPU(n_components=dim, opu_device=SimulatedOpuDevice(),
              max_n_features=dim)
    opu.fit1d(n_features=dim)

    rand_vec_1 = np.random.randn(dim)
    rand_vec_2 = np.random.randn(dim)
    alpha_1 = np.random.randn(1)
    alpha_2 = np.random.randn(1)
    lin_comb = alpha_1 * rand_vec_1 + alpha_2 * rand_vec_2
    res_lin_comb = enc_dec_fct(opu.linear_transform, lin_comb.reshape(1, -1))

    res_rand_vec_1 = enc_dec_fct(opu.linear_transform, rand_vec_1.reshape(1, -1))
    res_rand_vec_2 = enc_dec_fct(opu.linear_transform, rand_vec_2.reshape(1, -1))
    lin_comb_res = alpha_1 * res_rand_vec_1 + alpha_2 * res_rand_vec_2

    assert np.isclose(res_lin_comb, lin_comb_res, atol=1e-2).all()


def test_calibration_OPUFeatureMap(my_dim):
    sampling_method = "ARKM"
    sketch_dim = my_dim * 2
    Sigma = 0.876
    # Sigma = np.eye(my_dim) * Sigma
    nb_input = 10
    seed = 0
    # seed = np.random.randint(0, 2**10)

    for use_torch in [True]:
        opu = OPU(n_components=sketch_dim, opu_device=SimulatedOpuDevice(),
                  max_n_features=my_dim)
        opu.fit1d(n_features=my_dim)
        lst_omega = [sifact, _, R] = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                                   seed=seed, keep_splitted=True, return_torch=use_torch)
        lst_omega = list(lst_omega)
        OFM = OPUFeatureMap(f="complexexponential",
                                dimension=my_dim, SigFact=sifact, R=R,
                                opu=opu,
                                calibration_param_estimation=True,
                                calibration_forward=True,
                                calibration_backward=True,
                                calibrate_always=True,
                                re_center_result=False,
                            )
        directions = OFM.directions_matrix()
        lst_omega[1] = directions

        MFM = MatrixFeatureMap("complexexponential", tuple(lst_omega))
        input_mat = torch.from_numpy(np.random.randn(nb_input, my_dim))

        ofm_output = OFM(input_mat)
        mfm_output = MFM(input_mat)

        assert np.isclose(ofm_output, mfm_output).all()

    # input_vec = np.random.randn(1, my_dim)
    # ofm_output_grad = OFM.grad(input_vec)
    # mfm_output_grad = MFM.grad(input_vec)
    # assert np.isclose(ofm_output_grad, mfm_output_grad).all()


def test_OPUFeatureMap_multi_sigma(my_dim):
    print()
    for nb_sigmas in [0, 3, 1]:
        for nb_replicates in [2, 1]:
            print(f"nb_sigmas={nb_sigmas}")
            print(f"nb_replicates={nb_replicates}")
            sampling_method = "ARKM"
            sketch_dim = my_dim * 2
            Sigma = 0.876
            if nb_sigmas != 0:
                Sigma = np.array([np.abs(np.random.randn(1)) for _ in range(nb_sigmas)]).flatten()
            else:
                nb_sigmas = 1
            nb_input = 3
            seed = 0
            r_seeds = [seed] * nb_replicates

            opu = OPU(n_components=sketch_dim, opu_device=SimulatedOpuDevice(),
                      max_n_features=my_dim)
            opu.fit1d(n_features=my_dim)

            lst_omega = [sifact, _, R] = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                                                            seed=seed, keep_splitted=True, return_torch=True,
                                                                                            R_seeds=r_seeds)

            OFM = OPUFeatureMap(f="complexexponential",
                                    dimension=my_dim, SigFact=lst_omega[0], R=R,
                                    opu=opu,
                                    calibration_param_estimation=True,
                                    calibration_forward=True,
                                    calibration_backward=True,
                                    calibrate_always=True,
                                    re_center_result=False,
                                )
            # directions = OFM.directions_matrix()
            # lst_omega[1] = directions

            input_mat = np.random.randn(nb_input, my_dim)
            input_mat = torch.from_numpy(input_mat)

            ofm_output = OFM(input_mat)
            assert ofm_output.shape[-1] == nb_sigmas*sketch_dim*nb_replicates
            ofm_output = ofm_output.numpy()

            if nb_sigmas == 1:
                assert (np.tile(ofm_output[..., :sketch_dim], nb_sigmas*nb_replicates) == ofm_output).all()
            assert (np.tile(ofm_output[..., :sketch_dim*nb_sigmas], nb_replicates) == ofm_output).all()
