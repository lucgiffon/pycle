from abc import ABC, abstractmethod
from typing import Callable, Literal, Union, Optional, NoReturn, Tuple

import torch
import numpy as np
from pycle.sketching.feature_maps.non_linearities import _dico_nonlinearities_torch, _dico_normalization_rpf
from pycle.utils.encoding_decoding import enc_dec_fct, only_quantification_fct
from pycle.utils.intermediate_storage import IntermediateResultStorage


# cleaning make triple rademacher feature map
class FeatureMap(ABC):
    """
    Abstract feature map class
    Template for a generic Feature Map. Useful to check if an object is an instance of FeatureMap.
    """

    def __init__(self, f: Optional[Union[Literal["complexexponential", "universalquantization", "cosine", "none"], Callable]] = "complexexponential",
                 xi: Optional[torch.Tensor] = None, c_norm: Union[float, Literal["unit", "normalized"]] = 1.,
                 encoding_decoding: bool = False, quantification: bool = False, encoding_decoding_precision: int = 8,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float,
                 save_outputs: bool = False):
        """
        Parameters
        ----------
        f:
            The activation function for the feature map. Default: "complexexponential".
        xi:
            The dithering to add to the result of the random projection before the activation function.
            Default: no dithering.
        c_norm:
            Normalization factor for the feature map. Default: no normalization.
        encoding_decoding:
            Encode the input with `SeparatedBitPlans` before the feature map.
            Then decode the output accordingly.
        quantification:
            Quantify the input as if it was encoded decoded.
        encoding_decoding_precision:
            Maximum precision for the quantification and encoding_decoding parameters.
            Default: max possible precision.
        device:
            The device on which to perform the tensor operations. torch.device("cpu") or torch.device("cuda:*").
        dtype:
            The type of the tensor operations.
        save_outputs:
            Use the class `IntermediateResultStorage` to store the intermediate steps in the transformation by
            the feature map. Be careful with memory use. Default: False.
        """
        self.device = device
        self.dtype = dtype

        self.d, self._m = self.init_shape()

        self.name = None
        self.f = None
        self.update_activation(f)

        # 3) extract the dithering
        if xi is None:
            self.xi = torch.zeros(self._m).to(self.device)
            self.xi_all_zeros = True
        else:
            self.xi = xi.to(self.device)
            self.xi_all_zeros = all(self.xi == 0)

        # 4) extract the normalization constant
        if isinstance(c_norm, str):
            if c_norm.lower() in ['unit', 'normalized']:
                self.c_norm = 1. / np.sqrt(self._m)
            else:
                raise NotImplementedError("The provided c_norm name is not implemented.")
        else:
            self.c_norm = c_norm

        self.counter_call_sketching_operator = 0

        self.encoding_decoding = encoding_decoding
        self.quantification = quantification
        assert encoding_decoding is False or quantification is False, \
            "It is useless to do both quantification and encoding/decoding wrapping"
        self.encoding_decoding_precision = encoding_decoding_precision

        self.save_outputs = save_outputs

    def wrap_transform(self, transform: Callable, x: torch.Tensor, *args, **kwargs) -> Callable:
        """
        Wraps the `transform` function with encoding decoding or quantification utilities on the input `x`.

        It is usefull if the feature map is OPU based or for experiments.

        Parameters
        ----------
        transform:
            The transform method to call for x.
        x:
            The input to transform.
        args:
            List of positional arguments for the transform method.
        kwargs:
            Key value arguments for the transform method.

        Returns
        -------
            The output of the wrapped `transform` function applied on `x`.
        """
        if self.encoding_decoding:
            return lambda: enc_dec_fct(transform, x, *args, **kwargs)
        elif self.quantification:
            return lambda: only_quantification_fct(transform, x, *args, **kwargs)
        else:
            return lambda: transform(x)

    def account_call(self, x: torch.Tensor) -> NoReturn:
        """
        Increment the count of calls to the feature map.

        If `x` has ndim==2, then the counter of feature map calls is incremented as many times as there is rows in `x`.

        Parameters
        ----------
        x:
            The input to the feature map.
        """
        if len(x.shape) == 1:
            self.counter_call_sketching_operator += 1
        else:
            assert len(x.shape) == 2
            self.counter_call_sketching_operator += x.shape[0]

    def reset_counter(self) -> NoReturn:
        """
        Reset the counter of feature map calls to 0.
        """
        self.counter_call_sketching_operator = 0

    @abstractmethod
    def lin_op_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        The linear transformation (usually random projection) applied to the input before the non-linearity.

        Parameters
        ----------
        x:
            Input of the feature map.

        Returns
        -------
            The output of the linear transformation.
        """
        pass

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the feature map on `x`

        Parameters
        ----------
        x:
            The input to which the feature map must be applied.

        Returns
        -------
            The result of the feature map on `x`.
        """
        # first make the linear transformation
        out = self.lin_op_transform(x)
        if self.save_outputs:
            IntermediateResultStorage().add(out.cpu().numpy(), "output_y before non lin")

        # then apply the dithering and the activation function.
        if not self.xi_all_zeros:
            before_norm = self.f(out + self.xi)
        else:
            # small optimisation: if xi is full of zeros, it is useless to add it.
            before_norm = self.f(out)

        if self.save_outputs:
            IntermediateResultStorage().add(before_norm.cpu().numpy(), "output_y after non lin")

        # normalize the output of the feature map if necessary and return the result.
        if self.c_norm == 1.:
            # small optimisation: if the normalization is 1, it is useless to apply it.
            return before_norm
        else:
            return self.c_norm * before_norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the feature map on `x` and count the call.

        Parameters
        ----------
        x:
            The input to which the feature map must be applied.

        Returns
        -------
            The result of the feature map on `x`.

        """
        self.account_call(x)
        # this rpf dictionary use is here for legacy purposes: necessary for assymmetric compressive learning.
        return self.call(x) * (1. / _dico_normalization_rpf[self.name])

    @property
    def m(self) -> int:
        """
        Returns
        -------
            The number of output features of the feature map.
        """
        return self._m

    @abstractmethod
    def init_shape(self) -> Tuple[int, int]:
        """
        The shape of the linear transformation matrix used inside the feature map.

        The shape correspond to the (input, output) dimensions.

        Returns
        -------
            The (input, output) dimensions of the feature map.
        """
        pass

    def update_activation(self, f: Union[Callable, str]) -> NoReturn:
        """
        Update the activation function.

        Parameters
        ----------
        f: [callable, str]
            The new activation function.
        """
        if isinstance(f, str):
            try:
                self.name = f.lower()  # Keep the feature function name in memory so that we know we have a specific fct
                self.f = _dico_nonlinearities_torch[f.lower()]
            except KeyError:
                raise NotImplementedError(f"The provided feature map name {f} is not implemented.")
        elif callable(f):
            self.f = f
        else:
            raise ValueError(f"The provided feature map f={f} does not match any of the supported types.")
