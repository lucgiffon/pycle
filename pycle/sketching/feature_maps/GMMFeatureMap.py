import numpy as np

from pycle.sketching import fourierSketchOfGaussian, fourier_sketch_of_gaussianS
from pycle.utils import enc_dec_fct

from pycle.sketching.feature_maps.FeatureMap import FeatureMap
import torch


# schellekensvTODO find a better name
class GMMFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""

    def __init__(self, f, Omega, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        raise NotImplementedError("Class GMMFeatureMap is not available yet and must be adapted to torch.")
        self.Omega = Omega

        super().__init__(f=f, dtype=Omega.dtype, **kwargs)

        assert f == "None", "Only non-linearity f='None' is allowed in GMMFeatureMap"

    def init_shape(self):
        try:
            return self.Omega.shape
        except AttributeError:
            raise ValueError("The provided projection matrix Omega should be a (d,m) linear operator.")

    # call the FeatureMap object as a function
    def call(self, x):
        (mu, sig) = (x[..., :self.d], x[..., -self.d:])
        if x.ndim == 1:
            if not self.use_torch:
                mu = mu[np.newaxis, :]
                sig = sig[np.newaxis, :]
                # return fourierSketchOfGaussian(mu, np.diag(sig.squeeze()), self.Omega, self.xi, self.c_norm)
            else:
                mu = torch.unsqueeze(mu, 0)
                sig = sig.unsqueeze(sig, 0)

        return fourier_sketch_of_gaussianS(mu, sig, self.Omega, self.xi, self.c_norm, use_torch=self.use_torch)