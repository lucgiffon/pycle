from pycle.sketching.distribution_estimation import mu_estimation_ones, var_estimation_ones, var_estimation_randn, \
    var_estimation_any
from pycle.sketching.feature_maps.SimpleFeatureMap import SimpleFeatureMap
from lightonml import OPU
from lightonml.encoding.base import SeparatedBitPlanDecoder, SeparatedBitPlanEncoder
from lightonml.internal.simulated_device import SimulatedOpuDevice
import numpy as np
from scipy.linalg import hadamard
from fht import fht


# note that the encoder and decoder are instantitated here once and for all (and not at each function call)
def enc_dec_opu_transform(opu, x, precision_encoding=8):
    encoder = SeparatedBitPlanEncoder(precision=precision_encoding)
    # encoder = QuantizedSeparatedBitPlanEncoder(base=2, n_bits=precision_encoding)
    # x_enc = encoder.transform(x)
    x_enc = encoder.fit_transform(x)
    y_enc = opu.linear_transform(x_enc)
    decoder = SeparatedBitPlanDecoder(**encoder.get_params())
    # decoder = QuantizedMixingBitPlanDecoder(n_bits=precision_encoding, decoding_decay=2)
    y_dec = decoder.transform(y_enc)
    return y_dec


class OPUDistributionEstimator:
    def __init__(self, opu, input_dim, use_calibration=False):
        self.opu = opu
        self.d = input_dim
        self.m = self.opu.n_components

        self.use_calibration = use_calibration
        if self.use_calibration:
            self.FHB = self.calibrate_opu()
        else:
            self.FHB = None

    def OPU(self, x):
        return np.array(enc_dec_opu_transform(self.opu, x))

    def transform(self, x, direct=False):

        assert 0 < x.ndim <= 2
        if self.use_calibration:
            y = x @ self.FHB
        else:
            if direct:
                y = self.opu.linear_transform(x)
            else:
                if x.ndim == 1: x = x.reshape(1, -1)
                y = self.OPU(x)

        return y

    def calibrate_opu(self):
        H = hadamard(self.d)
        B = np.array((self.opu.linear_transform(H > 0) - self.opu.linear_transform(H < 0)))
        sqrt_d = np.sqrt(self.d)
        FHB = np.array([1. / self.d * fht(b) * sqrt_d for b in B.T]).T
        # FHB = H @ B / self.d
        return FHB

    def mu_estimation(self, method="ones"):
        if method == "ones":
            return self._mu_estimation_ones()
        elif method == "mean":
            try:
                return np.mean(self.FHB)
            except TypeError as e:
                raise ValueError(f"Method `mean` only works with `use_calibration` flag.")
        else:
            raise ValueError(f"Unknown method: {method}.")

    def _mu_estimation_ones(self):
        result = mu_estimation_ones(lambda x: self.transform(x, direct=True),
                                  self.d)
        # ones = np.ones(self.d)
        # if self.use_calibration:
        #     y = self.FHB @ ones
        # else:
        #     y = self.opu.linear_transform(ones)
        # result2 = np.sum(y) / (self.m * self.d)
        # assert result == result2
        return result

    def var_estimation(self, method="var", n_iter=1):
        if method == "ones":
            return self._var_estimation_ones()
        elif method == "any":
            return self._var_estimation_any(n_iter)
        elif method == "randn":
            return self._var_estimation_randn(n_iter)
        elif method == "var":
            try:
                return np.var(self.FHB)
            except TypeError:
                raise ValueError(f"Method `var` only works with `use_calibration` flag.")
        else:
            raise ValueError(f"Unknown method: {method}.")

    def _var_estimation_ones(self):
        result = var_estimation_ones(lambda x: self.transform(x, direct=True),
                                     self.d)
        # ones = np.ones(self.d)
        # if self.use_calibration:
        #     y = ones @ self.FHB
        # else:
        #     y = self.opu.linear_transform(ones)
        # D_var = np.var(y)
        # result2 = D_var / self.d
        # assert result == result2
        return result

    def _var_estimation_randn(self, n_iter=1):
        result = var_estimation_randn(lambda x: self.transform(x),
                                      self.d, n_iter)

        # mu = self._mu_estimation_ones()
        # x = np.random.randn(n_iter, self.d)
        #
        # if self.use_calibration:
        #     y = x @ self.FHB
        # else:
        #     y = self.OPU(x)

        # D_var_plus_mu = np.var(y)
        # var = D_var_plus_mu / self.d - (mu ** 2)

        # assert np.isclose(var, result)
        return result

    def _var_estimation_any(self, n_iter=1):
        result = var_estimation_any(lambda x: self.transform(x),
                                    self.d, n_iter)
        # # only works if mu is zero
        # X = np.random.rand(n_iter, self.d)
        # X_norm_2 = np.linalg.norm(X, axis=1).reshape(n_iter, -1)
        # X /= X_norm_2
        #
        # if self.use_calibration:
        #     Y = X @ self.FHB
        # else:
        #     Y = self.OPU(X)
        #
        # Y_norm_2 = np.linalg.norm(Y) ** 2
        # var = Y_norm_2 / Y.size
        # assert var == result
        return result


# schellekensvTODO find a better name
class OPUFeatureMap(SimpleFeatureMap):
    """Feature map the type Phi(x) = c_norm*f(OPU(x) + xi)."""

    def __init__(self, f, opu, Sigma=None, light_memory=True, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        # todoopu initialiser l'opu
        self.opu = opu
        self.light_memory = light_memory

        super().__init__(f, **kwargs)
        self.distribution_estimator = OPUDistributionEstimator(self.opu, self.d, use_calibration=not light_memory)

        self.mu_opu, self.std_opu, self.norm_scaling = self.get_distribution_opu()
        # self.mu_opu, self.std_opu, self.norm_scaling = self.get_distribution_opu(light_memory=False)

        # todoopu deplacer ca
        self.Sigma = Sigma
        if self.Sigma is None:
            self.Sigma = np.identity(self.opu.max_n_features)  # todoopu attention a ca car ce n'est disponible qu'en mode simulé
        self.SigFact = np.linalg.inv(np.linalg.cholesky(self.Sigma))
        self.R = np.abs(np.random.randn(self.m))  # folded standard normal distribution radii
        # self.norm_scaling = np.sqrt(self.d) #* np.ones(self.m)

    def get_distribution_opu(self):
        if not self.light_memory:
            mu = self.distribution_estimator.mu_estimation(method="mean")
            std = np.sqrt(self.distribution_estimator.var_estimation(method="var"))
            col_norm = np.linalg.norm(self.distribution_estimator.opu, axis=0)

        else:
            # todo choisir le n_iter dynamiquement
            mu = self.distribution_estimator.mu_estimation(method="ones")
            std = np.sqrt(self.distribution_estimator.var_estimation(method="ones"))
            col_norm = np.sqrt(self.d) * np.ones(self.m)

        return mu, std, col_norm

    def init_shape(self):
        if isinstance(self.opu.device, SimulatedOpuDevice):
            return self.opu.max_n_features, self.opu.n_components
        else:
            return None, self.opu.n_components

    def _OPU(self, x):
        return enc_dec_opu_transform(self.opu, x)

    def applyOPU(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x = x @ self.SigFact

        # x_enc = self.encoder.transform(x)
        # y_enc = self.opu.transform(x_enc)
        # # now center the coefficients of the matrix then scale the result
        # # mu_x_enc = self.mu_opu * np.sum(x_enc, axis=1)  # sum other all dims
        # # y_enc = y_enc - mu_x_enc.reshape(-1, 1)
        # # y_enc = self.R * y_enc * 1./self.std_opu * 1./self.norm_scaling
        # y_dec = self.decoder.transform(y_enc)
        y_dec = self._OPU(x)

        # now center the coefficients of the matrix then scale the result
        mu_x = self.mu_opu * np.sum(x, axis=1)  # sum other all dims
        y_dec = y_dec - mu_x.reshape(-1, 1)
        y_dec = self.R * y_dec * 1./self.std_opu * 1./self.norm_scaling

        out = y_dec
        return out

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        return self.c_norm * self.f(self.applyOPU(x) + self.xi)  # Evaluate the feature map at x

    def grad(self, x):
        raise NotImplementedError("OPU doesn't have a gradient.")


if __name__ == "__main__":
    opu = OPU(n_components=2, opu_device=SimulatedOpuDevice(),
              max_n_features=2)
    opu.fit1d(n_features=2)
    # Phi = OPUFeatureMap("ComplexExponential", opu)
    sample = np.random.rand(1, 2)
    enc_dec_opu_transform(opu, sample)
    enc_dec_opu_transform(opu, sample)

    opudistestim = OPUDistributionEstimator(opu, 2)
    opudistestim.mu_estimation("ones")
    opudistestim.var_estimation("ones")
    opudistestim.var_estimation("any")
    opudistestim.var_estimation("randn")
    opudistestim = OPUDistributionEstimator(opu, 2, use_calibration=True)
    opudistestim.mu_estimation("ones")
    opudistestim.var_estimation("ones")
    opudistestim.var_estimation("any")
    opudistestim.var_estimation("randn")
    opudistestim.var_estimation("var")
