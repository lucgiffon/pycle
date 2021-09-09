import pytest
import numpy as np
import pycle.sketching.frequency_sampling

from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
from pycle.sketching.feature_maps.MatrixFeatureMap import MatrixFeatureMap
from pycle.sketching.feature_maps.OPUFeatureMap import calibrate_lin_op, enc_dec_fct, OPUFeatureMap


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
    Sigma = np.eye(my_dim) * 0.876
    nb_input = 10
    seed = 0
    # seed = np.random.randint(0, 2**10)

    opu = OPU(n_components=sketch_dim, opu_device=SimulatedOpuDevice(),
              max_n_features=my_dim)
    opu.fit1d(n_features=my_dim)
    OFM = OPUFeatureMap(f="ComplexExponential",
                            dimension=my_dim,
                            opu=opu,
                            Sigma=Sigma,
                            calibration_param_estimation=True,
                            calibration_forward=True,
                            calibration_backward=True,
                            calibrate_always=True,
                            re_center_result=False,
                            sampling_method=sampling_method,
                        seed=seed)

    randn_opu_matrix_0_1 = OFM.get_randn_mat()
    Omega = pycle.sketching.frequency_sampling.drawFrequencies(sampling_method, my_dim, sketch_dim, Sigma,
                                                               randn_mat_0_1=randn_opu_matrix_0_1, seed=seed)
    MFM = MatrixFeatureMap("ComplexExponential", Omega)

    input_mat = np.random.randn(nb_input, my_dim)

    ofm_output = OFM(input_mat)
    mfm_output = MFM(input_mat)

    assert np.isclose(ofm_output, mfm_output).all()

    input_vec = np.random.randn(1, my_dim)
    ofm_output_grad = OFM.grad(input_vec)
    mfm_output_grad = MFM.grad(input_vec)
    assert np.isclose(ofm_output_grad, mfm_output_grad).all()
