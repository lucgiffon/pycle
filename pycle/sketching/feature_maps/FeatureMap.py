from abc import ABC, abstractmethod

import torch
import numpy as np
from pycle.sketching.feature_maps import _dico_nonlinearities, _dico_nonlinearities_torch
from pycle.utils import enc_dec_fct, only_quantification_fct


class FeatureMap(ABC):
    """Abstract feature map class
    Template for a generic Feature Map. Useful to check if an object is an instance of FeatureMap."""

    def __init__(self, *args, f="complexexponential", xi=None, c_norm=1., encoding_decoding=False,
                 quantification=False, encoding_decoding_precision=8, use_torch=False, device=None, dtype=None, **kwargs):
        """
        - f can be one of the following:
            -- a string for one of the predefined feature maps:
                -- "complexExponential"
                -- "universalQuantization"
                -- "cosine"
            -- a callable function
            -- a tuple of function (specify the derivative too)

        """
        self.use_torch = use_torch
        self.device = device
        self.dtype = dtype

        if use_torch:
            self.module_math_functions = torch
            self.dico_nonlinearities = _dico_nonlinearities_torch
        else:
            self.module_math_functions = np
            self.dico_nonlinearities = _dico_nonlinearities

        self.d, self._m = self.init_shape()

        self.name = None
        if isinstance(f, str):
            try:
                (self.f, self.f_grad) = self.dico_nonlinearities[f.lower()]
                self.name = f  # Keep the feature function name in memory so that we know we have a specific fct
            except KeyError:
                raise NotImplementedError(f"The provided feature map name {f} is not implemented with `use_torch`={use_torch}.")
        elif callable(f):
            (self.f, self.f_grad) = (f, None)
        elif (isinstance(f, tuple)) and (len(f) == 2) and (callable(f[0]) and callable(f[1])):
            (self.f, self.f_grad) = f
        else:
            raise ValueError("The provided feature map f does not match any of the supported types.")

        # 3) extract the dithering
        if xi is None:
            self.xi = self.module_math_functions.zeros(self._m)
        else:
            self.xi = xi

        # 4) extract the normalization constant
        if isinstance(c_norm, str):
            if c_norm.lower() in ['unit', 'normalized']:
                self.c_norm = 1. / self.module_math_functions.sqrt(self._m)
            else:
                raise NotImplementedError("The provided c_norm name is not implemented.")
        else:
            self.c_norm = c_norm

        self.counter_call_sketching_operator = 0

        self.encoding_decoding = encoding_decoding
        self.quantification = quantification
        assert encoding_decoding is False or quantification is False
        self.encoding_decoding_precision = encoding_decoding_precision

    def wrap_transform(self, transform, x, *args, **kwargs):
        if self.encoding_decoding:
            return lambda: enc_dec_fct(transform, x, *args, **kwargs)
        elif self.quantification:
            return lambda: only_quantification_fct(transform, x, *args, **kwargs)
        else:
            return lambda: transform(x)

    def account_call(self, x):
        if len(x.shape) == 1:
            self.counter_call_sketching_operator += 1
        else:
            assert len(x.shape) == 2
            self.counter_call_sketching_operator += x.shape[0]

    def reset_counter(self):
        self.counter_call_sketching_operator = 0

    @abstractmethod
    def call(self, x):
        raise NotImplementedError("The way to compute the feature map is not specified.")

    def __call__(self, x):
        self.account_call(x)
        return self.call(x)

    @abstractmethod
    def grad(self, x):
        raise NotImplementedError("The way to compute the gradient of the feature map is not specified.")

    @property
    def m(self):
        return self._m

    @abstractmethod
    def init_shape(self):
        pass
