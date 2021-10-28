import numpy as np
from pycle.utils import enc_dec_fct, LinearFunctionEncDec

from pycle.sketching.feature_maps.FeatureMap import FeatureMap


# schellekensvTODO find a better name
class MatrixFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        self.Omega = Omega
        super().__init__(f, dtype=Omega.dtype, **kwargs)

    def init_shape(self):
        try:
            return self.Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    def _apply_mat(self, x):
        if self.use_torch:
            if x.ndim == 1:
                return LinearFunctionEncDec.apply(x.unsqueeze(0), self.Omega, self.quantification, self.encoding_decoding).squeeze(0)
            else:
                return LinearFunctionEncDec.apply(x, self.Omega, self.quantification,
                                                  self.encoding_decoding)
        else:
            return self.wrap_transform(lambda inp: inp @ self.Omega, x, precision_encoding=self.encoding_decoding_precision)()

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(self._apply_mat(x) + self.xi)  # Evaluate the feature map at x

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