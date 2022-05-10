from pycle.utils import is_number
import numpy as np


def test_is_number():
    assert is_number(1.8) == True
    assert is_number(np.float16(1.8)) == True
    assert is_number(1) == True
    assert is_number("1") == False
    assert is_number("1.2") == False
    assert is_number(np.array([[1.2]])) == False
    assert is_number(np.array([[1.2, 3]])) == False


