from typing import Literal, Union, Callable, Optional, NoReturn, Tuple

import numpy as np
import torch
from enum import Enum

from pycle.sketching.frequency_sampling import rebuild_Omega_from_sig_dir_R
from pycle.utils import LinearFunctionEncDec, MultiSigmaARFrequencyMatrixLinApEncDec, is_number

from pycle.sketching.feature_maps.FeatureMap import FeatureMap


# schellekensvTODO find a better name
from pycle.utils.optim import IntermediateResultStorage


class MatrixFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f: Optional[Union[Literal["complexexponential", "universalquantization", "cosine"], Callable]],
                 Omega: [torch.Tensor, tuple, list], device: torch.device = torch.device("cpu"), **kwargs):
        """
        Parameters
        ----------
        f:
            The activation function for the feature map. Default: "complexexponential".
        Omega:
            The random projection matrix. If it is a tuple or a list,
        device:
            The device on which to perform the tensor operations. torch.device("cpu") or torch.device("cuda:*").
        kwargs:
            Other key word arguments for FeatureMap object.
        """
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        if type(Omega) == tuple or type(Omega) == list:
            self.splitted_Omega = True
            # (sigma, directions, amplitudes)
            self.SigFact = Omega[0].to(device)
            self.directions = Omega[1].to(device)
            self.R = Omega[2].to(device)
            if self.R.ndim == 1:
                self.R = self.R.unsqueeze(-1)

            assert self.R.shape[0] == self.directions.shape[1]
            if is_number(self.SigFact):
                self.SigFact = torch.Tensor([self.SigFact]).to(device)

            self.bool_sigfact_a_matrix = self.SigFact.ndim > 1
        else:
            self._Omega = Omega.to(device)
            self.splitted_Omega = False

        super().__init__(f, dtype=self.Omega_dtype, device=device, **kwargs)

    @property
    def Omega_dtype(self) -> torch.dtype:
        """
        Returns
        -------
            The type of the Omega linear transform matrix
        """
        if self.splitted_Omega:
            return torch.promote_types(torch.promote_types(self.directions.dtype, self.R.dtype), self.SigFact.dtype)
        else:
            return self._Omega.dtype

    @property
    def Omega(self) -> torch.Tensor:
        """
        Be careful with the memory use if the `splitted_Omega` attribute is True.

        Returns
        -------
            The (reconstructed) Omega matrix.
        """
        if self.splitted_Omega:
            if self.bool_sigfact_a_matrix:
                return self.SigFact @ self.directions * self.R
            elif not is_number(self.SigFact):
                return rebuild_Omega_from_sig_dir_R(self.SigFact, self.directions, self.R)
            else:
                return self.SigFact * self.directions * self.R
        else:
            return self._Omega

    def unsplit(self) -> NoReturn:
        """
        Rebuild the Omega matrix and remove the parts.
        """
        assert self.splitted_Omega == True
        self._Omega = self.Omega
        self.splitted_Omega = False
        del self.SigFact
        del self.directions
        del self.R

    def init_shape(self) -> Tuple[int, int]:
        """
        The shape of the linear transformation matrix used inside the feature map.

        The shape correspond to the (input, output) dimensions.

        Returns
        -------
            The (input, output) dimensions of the feature map.
        """
        try:
            if self.splitted_Omega:
                return (self.directions.shape[0],
                        self.directions.shape[1] * len(self.SigFact) * self.R.shape[-1])
            else:
                return self._Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    def _apply_mat(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x:
            Input of the feature map.

        Returns
        -------
            The output of the Omega matrix applied to the input.
        """

        # the rest of the function only works for data presented as rows. If x is a column vector, make it a row.
        unsqueezed = False
        if x.ndim == 1:
            unsqueezed = True
            x = x.unsqueeze(0)

        if self.save_outputs:
            IntermediateResultStorage().add(x.cpu().numpy(), "input_x")

        if self.splitted_Omega:
            result = MultiSigmaARFrequencyMatrixLinApEncDec.apply(x, self.SigFact, self.directions, self.R,
                                                                  self.quantification, self.encoding_decoding, self.encoding_decoding_precision,
                                                                  self.save_outputs)
        else:
            result = LinearFunctionEncDec.apply(x, self.Omega,
                                                self.quantification, self.encoding_decoding, self.encoding_decoding_precision)

        if self.save_outputs:
            IntermediateResultStorage().add(result.cpu().numpy(), "output_y")

        # return a column vector if the input was a column.
        if unsqueezed:
            return result.squeeze(0)
        else:
            return result

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
        return self._apply_mat(x)
