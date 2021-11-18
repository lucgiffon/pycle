import numpy as np
import numbers
import torch
from pycle.utils import enc_dec_fct, LinearFunctionEncDec, MultiSigmaARFrequencyMatrixLinApEncDec

from pycle.sketching.feature_maps.FeatureMap import FeatureMap


# schellekensvTODO find a better name
class MatrixFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform

        if type(Omega) == tuple or type(Omega) == list:
            self.splitted_Omega = True
            # (sigma, directions, amplitudes)
            self.SigFact = Omega[0]
            self.bool_sigfact_a_matrix = (isinstance(self.SigFact, torch.Tensor) or isinstance(self.SigFact,
                                                                                               np.ndarray)) and self.SigFact.ndim > 1
            self.bool_multiple_sigmas = (isinstance(self.SigFact, np.ndarray) or isinstance(self.SigFact, torch.Tensor)) and self.SigFact.ndim == 1 and len(self.SigFact) > 1
            assert self.bool_sigfact_a_matrix or self.bool_multiple_sigmas or isinstance(self.SigFact, numbers.Number) or len(self.SigFact) == 1
            self.directions = Omega[1]
            self.R = Omega[2]
        else:
            self._Omega = Omega
            self.splitted_Omega = False
        super().__init__(f, dtype=self.Omega_dtype, **kwargs)

    def Omega_dtype(self):
        if self.splitted_Omega:
            promote_types = self.module_math_functions.promote_types
            return promote_types(promote_types(self.directions.dtype, self.R.dtype), self.SigFact.dtype)
        else:
            return self._Omega.dtype

    @property
    def Omega(self):
        if self.splitted_Omega:
            if self.bool_sigfact_a_matrix:
                return self.SigFact @ self.directions * self.R
            elif self.bool_multiple_sigmas:
                raise ValueError(" This happens because use_torch == False:"
                                 " Property Omega shouldn't be used when bool_multiple_sigmas is True"
                                 " because it involves to reconstruct the full matrix which makes"
                                 " the multiple sigmas option useless.")
                dr = self.directions * self.R
                r = self.module_math_functions.einsum("ij,jkl->kil", self.SigFact[:, self.module_math_functions.newaxis], dr[self.module_math_functions.newaxis])
                return r.reshape(self.d, self.m)
            else:
                return self.SigFact * self.directions * self.R
        else:
            return self._Omega

    def unsplit(self):
        assert self.splitted_Omega == True
        self._Omega = self.Omega
        self.splitted_Omega = False

        del self.SigFact
        del self.directions
        del self.R

    def init_shape(self):
        try:
            if self.splitted_Omega:
                if self.bool_multiple_sigmas:
                    return (self.directions.shape[0],
                                self.directions.shape[1] * len(self.SigFact))

                else:
                    return self.directions.shape
            else:
                return self._Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    def _apply_mat(self, x):
        if self.use_torch:
            if x.ndim == 1:
                if self.bool_multiple_sigmas:
                    return MultiSigmaARFrequencyMatrixLinApEncDec.apply(x.unsqueeze(0), self.SigFact, self.directions, self.R, self.quantification, self.encoding_decoding, self.encoding_decoding_precision).squeeze(0)
                else:
                    return LinearFunctionEncDec.apply(x.unsqueeze(0), self.Omega, self.quantification, self.encoding_decoding, self.encoding_decoding_precision).squeeze(0)
            else:
                if self.bool_multiple_sigmas:
                    return MultiSigmaARFrequencyMatrixLinApEncDec.apply(x, self.SigFact, self.directions, self.R, self.quantification, self.encoding_decoding, self.encoding_decoding_precision)
                else:
                    return LinearFunctionEncDec.apply(x, self.Omega, self.quantification,
                                                  self.encoding_decoding, self.encoding_decoding_precision)
        else:
            return self.wrap_transform(lambda inp: inp @ self.Omega, x, precision_encoding=self.encoding_decoding_precision)()

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        if self.bool_multiple_sigmas:
            return self.c_norm * self.f(self._apply_mat(x) + self.xi)  # Evaluate the feature map at x
        else:
            return self.c_norm * self.f(self._apply_mat(x) + self.xi)  # Evaluate the feature map at x

        # return self.c_norm * self.f(self._apply_mat(x) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        """Gradient (Jacobian matrix) of Phi, as a (n_x,d,m)-numpy array. n_x being the batch size of x."""
        if self.use_torch:
            raise NotImplementedError("No gradient available with `use_torch`=True")
        else:
            #return self.c_norm * self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi) * self.Omega
            f_grad_val = self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi)
            new = self.c_norm * np.einsum("ij,kj->ikj", f_grad_val, self.Omega)
            if x.shape[0] == 1 or x.ndim == 1:
                return new.squeeze()
            else:
                return new