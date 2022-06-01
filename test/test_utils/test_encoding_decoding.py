from pycle.utils.encoding_decoding import enc_dec_fct, only_quantification_fct
import numpy as np
import torch
from lightonml import OPU
import pytest
from lightonml.internal.simulated_device import SimulatedOpuDevice


@pytest.fixture
def dim():
    return 10


@pytest.fixture
def opu(dim):
    opu_ = OPU(n_components=10, max_n_features=10,  opu_device=SimulatedOpuDevice())
    opu_.fit1d(n_features=dim)
    return opu_


def test_enc_dec_fct(opu, dim):
    numpy_mat = np.random.randn(dim, dim)
    numpy_x = np.random.randn(1, 10)
    enc_dec_fct(lambda x: x @ numpy_mat, numpy_x)

    enc_dec_fct(opu.linear_transform, numpy_x)

    torch_mat = torch.from_numpy(numpy_mat).to(torch.float)
    torch_x = torch.from_numpy(numpy_x)
    # after encoding, x become of type Byte which is not compatible with torch.float
    # so casting x to float is necessary...
    enc_dec_fct(lambda x: x.to(torch.float) @ torch_mat, torch_x)

    enc_dec_fct(opu.linear_transform, torch_x)


def test_enc_dec_fct_no_change(opu, dim):
    # if there is only zero and ones in the input, there shouldn't be any noise brought by encoding decoding
    x = np.random.choice([0, 1], (1, dim))
    output_normal = opu.linear_transform(x)
    output_enc_dec = enc_dec_fct(opu.linear_transform, x)
    assert np.isclose(output_normal, output_enc_dec).all()


def test_only_quantification_fct(opu, dim):
    numpy_mat = np.random.randn(dim, dim)
    numpy_x = np.random.randn(1, 10)
    only_quantification_fct(lambda x: x @ numpy_mat, numpy_x)

    only_quantification_fct(opu.linear_transform, numpy_x)

    torch_mat = torch.from_numpy(numpy_mat).to(torch.float)
    torch_x = torch.from_numpy(numpy_x)
    # after encoding, x become of type Byte which is not compatible with torch.float
    # so casting x to float is necessary...
    only_quantification_fct(lambda x: x.to(torch.float) @ torch_mat, torch_x)

    only_quantification_fct(opu.linear_transform, torch_x)


def test_only_quantification_fct_no_change(opu, dim):
    # if there is only zero and ones in the input, there shouldn't be any noise brought by encoding decoding
    x = np.random.choice([0, 1], (1, dim))
    output_normal = opu.linear_transform(x)
    output_enc_dec = only_quantification_fct(opu.linear_transform, x)
    assert (output_normal == output_enc_dec).all()


