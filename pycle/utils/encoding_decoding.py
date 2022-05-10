"""
This module contains encoding/decoding utilities necessary when working with the OPU function.

The OPU function only takes inputs with values in {0, 1} so all OPU calls must be wrapped with
separatedbitplanencoding functions (provided by lighton).
"""

from typing import Callable, Union

import numpy as np
import torch
from lightonml.encoding.base import SeparatedBitPlanEncoder, SeparatedBitPlanDecoder


def enc_dec_fct(fct: Callable, x: Union[torch.Tensor, np.ndarray], precision_encoding: int = 6):
    """
    Encode x as bit planes for linear transformation then decode the result.
    This function just makes a transformation of x after encoding it. Then return the decoded result.

    Parameters
    ----------
    fct:
        fct taking x as input and needing encoding/decoding wrapping.
    x:
        The input to transform. It must have len(shape)==2 with single observations as rows. If there is only
        one observation to transform, this observation must be formated as a row.
    precision_encoding:
        Encoding precision in number of bits for quantification. It is useless to set it to a value greater than 6.

    Returns
    -------
        The transformed input decoded(fct(encoded(x))).
    """
    encoder = SeparatedBitPlanEncoder(precision=precision_encoding)
    x_enc = encoder.fit_transform(x)
    y_enc = fct(x_enc)
    decoder = SeparatedBitPlanDecoder(**encoder.get_params())
    y_dec = decoder.transform(y_enc)
    return y_dec


def only_quantification_fct(fct: Callable, x: Union[torch.Tensor, np.ndarray], precision_encoding=8):
    """
    Encode x as bit planes then decode it directly before the fct call.

    This function just makes a transformation of x after an encoding/decoding (noisy) step.

    Input is encoded.
    Then decoded.
    Then transformed.
    The result is returned.

    The idea is just to simulate the encoding/decoding noise.

    Parameters
    ----------
    fct:
        fct taking x as input and needing encoding/decoding wrapping.
    x:
        The input to transform.
    precision_encoding:
        Encoding precision in number of bits for quantification. It is useless to set it to a value greater than 6.

    Returns
    -------
        The transformed input fct(decoded(encoded(x))).

    """
    encoder = SeparatedBitPlanEncoder(precision=precision_encoding)
    x_enc = encoder.fit_transform(x)
    decoder = SeparatedBitPlanDecoder(**encoder.get_params())
    x_dec = decoder.transform(x_enc)
    y_dec = fct(x_dec)
    return y_dec
