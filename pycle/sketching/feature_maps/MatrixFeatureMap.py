import numpy as np
from pycle.sketching.feature_maps.OPUFeatureMap import enc_dec_fct

from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap


# schellekensvTODO find a better name
class MatrixFeatureMap(SimpleFeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        self.Omega = Omega
        super().__init__(f, **kwargs)

    def init_shape(self):
        try:
            return self.Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    def _apply_mat(self, x):
        if self.encoding_decoding:
            return enc_dec_fct(lambda inp: np.matmul(inp, self.Omega), x, precision_encoding=self.encoding_decoding_precision)
        else:
            return np.matmul(x, self.Omega)

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(self._apply_mat(x) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        """Gradient (Jacobian matrix) of Phi, as a (n_x,d,m)-numpy array. n_x being the batch size of x."""
        #return self.c_norm * self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi) * self.Omega
        f_grad_val = self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi)
        new = self.c_norm * np.einsum("ij,kj->ikj", f_grad_val, self.Omega)
        if x.shape[0] == 1 or x.ndim == 1:
            return new.squeeze()
        else:
            return new