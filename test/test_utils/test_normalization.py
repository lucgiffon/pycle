import numpy as np
import torch

from pycle.utils.normalization import get_backend_from_object, get_normalization_factor_from_string


def test_get_backend_from_object():
    obj = np.random.randn(10)
    assert get_backend_from_object(obj) == np

    obj = torch.rand(10)
    assert get_backend_from_object(obj) == torch


def test_get_normalization_factor_from_string_linf_unit_ball():
    obj = np.random.randn(10, 20)
    linf_normalization = (np.max(np.absolute(obj)) + 1e-6)

    assert (float(get_normalization_factor_from_string(obj, 'l_inf-unit-ball')) == float(linf_normalization))
    assert (float(get_normalization_factor_from_string(torch.from_numpy(obj), 'l_inf-unit-ball')) == float(linf_normalization))


def test_get_normalization_factor_from_string_l2_unit_ball():
    obj = np.random.randn(10, 20)
    l2_normalization = (np.max(np.linalg.norm(obj, axis=1)) + 1e-6)

    assert np.isclose(float(get_normalization_factor_from_string(obj, 'l_2-unit-ball')), float(l2_normalization))
    assert np.isclose(float(get_normalization_factor_from_string(torch.from_numpy(obj), 'l_2-unit-ball')), float(l2_normalization))

