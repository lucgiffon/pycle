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
        self.Omega = Omega
        assert self.use_torch == True, "GMMFeature map only available with torch (gradient not implemented and maybe other bugs)"
        assert f == "None", "Only non-linearity f='None' is allowed in GMMFeatureMap"
        super().__init__(f, **kwargs)

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

    def grad(self, x):
        """Gradient (Jacobian matrix) of Phi, as a (n_x,d,m)-numpy array. n_x being the batch size of x."""
        if self.use_torch:
            raise NotImplementedError("No gradient available with `use_torch`=True")
        else:
            raise NotImplementedError("Gradient computation for GMMFeatureMap doesn't work.")
            (mu, sig) = (x[..., :self.d], x[..., -self.d:])

            sketch_of_atom = fourierSketchOfGaussian(mu, np.diag(sig.squeeze()), self.Omega, self.xi, self.c_norm)

            jacobian = 1j * np.zeros((self.d*2, self.m))
            jacobian[:self.d] = 1j * self.Omega * sketch_of_atom  # Jacobian w.r.t. mu
            jacobian[self.d:] = -0.5 * (self.Omega ** 2) * sketch_of_atom  # Jacobian w.r.t. sigma

            return jacobian
