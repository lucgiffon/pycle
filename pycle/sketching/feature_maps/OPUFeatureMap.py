import torch

from pycle.sketching import FeatureMap
from pycle.sketching.distribution_estimation import mu_estimation_ones, var_estimation_ones, var_estimation_randn, \
    var_estimation_any
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
import numpy as np
from pycle.sketching.frequency_sampling import sampleFromPDF, pdfAdaptedRadius
from pycle.utils import enc_dec_fct, LinearFunctionEncDec, OPUFunctionEncDec
from scipy.linalg import hadamard
from fht import fht


def calibrate_lin_op(fct_lin_op, dim):
    first_pow_of_2_gt_d = 2 ** int(np.ceil(np.log2(dim)))
    H = hadamard(first_pow_of_2_gt_d)
    H_truncated_left = H[:dim, :dim]
    H_truncated_right = H[:dim, dim:]
    B_truncated_left = np.array(
        fct_lin_op(H_truncated_left > 0) - fct_lin_op(H_truncated_left < 0))
    B_truncated_right = np.array(
        fct_lin_op(H_truncated_right.T > 0) - fct_lin_op(H_truncated_right.T < 0))
    B = np.vstack([B_truncated_left, B_truncated_right])
    # B = np.array((self.opu.linear_transform(H > 0) - self.opu.linear_transform(H < 0)))
    sqrt_d = np.sqrt(first_pow_of_2_gt_d)
    FHB = np.array([1. / first_pow_of_2_gt_d * fht(b) * sqrt_d for b in B.T]).T
    # FHB = H @ B / self.d
    return FHB[:dim]


class OPUDistributionEstimator:
    def __init__(self, opu, input_dim, compute_calibration=False, use_calibration_transform=False, encoding_decoding_precision=8, use_torch=False):
        """

        :param opu:
        :param input_dim:
        :param use_calibration: If False, lighter memory but parameter estimation is less accurate.
        """
        self.opu = opu
        self.d = input_dim
        self.m = self.opu.n_components

        self.use_torch = use_torch
        if use_torch:
            self.module_math_functions = torch
        else:
            self.module_math_functions = np

        self.use_calibration_transform = use_calibration_transform
        self.compute_calibration = compute_calibration
        assert use_calibration_transform is False or compute_calibration is True
        if self.compute_calibration:
            self.FHB = self.calibrate_opu()
        else:
            self.FHB = None

        self.encoding_decoding_precision = encoding_decoding_precision

    def OPU(self, x):
        if self.use_torch:
            return torch.from_numpy(enc_dec_fct(self.opu.linear_transform, x, precision_encoding=self.encoding_decoding_precision))
        else:
            return np.array(enc_dec_fct(self.opu.linear_transform, x, precision_encoding=self.encoding_decoding_precision))

    def transform(self, x, direct=False):
        """

        :param x:
        :param direct: if True, transforms x without encoding/decoding wrapping.
        :return:
        """
        assert 0 < x.ndim <= 2
        if self.use_calibration_transform:
            y = x @ self.FHB
        else:
            if direct:
                y = self.opu.linear_transform(x)
            else:
                if x.ndim == 1: x = x.reshape(1, -1)
                y = self.OPU(x)

        return y

    def calibrate_opu(self):
        if self.use_torch:
            return torch.from_numpy(calibrate_lin_op(lambda x: self.opu.linear_transform(x), self.d))
        else:
            return calibrate_lin_op(lambda x: self.opu.linear_transform(x), self.d)

    def mu_estimation(self, method="ones"):
        """

        :param method: Choose in ["ones", "mean"]
        :return:
        """
        if method == "ones":
            return self._mu_estimation_ones()
        elif method == "mean":
            try:
                return self.module_math_functions.mean(self.FHB)
            except TypeError as e:
                raise ValueError(f"Method `mean` only works with `compute_calibration` flag.")
        else:
            raise ValueError(f"Unknown method: {method}.")

    def _mu_estimation_ones(self):
        result = mu_estimation_ones(lambda x: self.transform(x, direct=True),
                                  self.d)
        return result

    def var_estimation(self, method="var", n_iter=1):
        """

        :param method: Choose in ["ones", "any", "randn", "var"]
        :param n_iter: Number of iterations for methods "any" and "randn". No effect otherwise.
        :return:
        """
        if method == "ones":
            return self._var_estimation_ones()
        elif method == "any":
            return self._var_estimation_any(n_iter)
        elif method == "randn":
            return self._var_estimation_randn(n_iter)
        elif method == "var":
            try:
                return self.module_math_functions.var(self.FHB)
            except TypeError:
                raise ValueError(f"Method `var` only works with `compute_calibration` flag.")
        else:
            raise ValueError(f"Unknown method: {method}.")

    def _var_estimation_ones(self):
        result = var_estimation_ones(lambda x: self.transform(x, direct=True),
                                     self.d)
        return result

    def _var_estimation_randn(self, n_iter=1):
        result = var_estimation_randn(lambda x: self.transform(x),
                                      self.d, n_iter)
        return result

    def _var_estimation_any(self, n_iter=1):
        result = var_estimation_any(lambda x: self.transform(x),
                                    self.d, n_iter)
        return result


class OPUFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(OPU(x) + xi)."""

    def __init__(self, f, opu, SigFact, R, dimension=None, calibration_param_estimation=False, calibration_forward=False, calibration_backward=False, calibrate_always=False, re_center_result=False, **kwargs):
        # 2) extract Omega the projection matrix schellekensvTODO allow callable Omega for fast transform
        # todoopu initialiser l'opu
        self.opu = opu
        self.provided_dimension = dimension

        self.SigFact = SigFact
        self.bool_sigfact_a_matrix = (isinstance(self.SigFact, torch.Tensor) or isinstance(self.SigFact, np.ndarray)) and self.SigFact.ndim > 1
        self.bool_multiple_sigmas = (isinstance(self.SigFact, torch.Tensor) or isinstance(self.SigFact, np.ndarray)) and self.SigFact.ndim == 1
        self._Omega = None
        self.R = R

        super().__init__(f, **kwargs)

        self.light_memory = (not (calibration_param_estimation or calibration_forward or calibration_backward)) and (not calibrate_always)
        self.distribution_estimator = OPUDistributionEstimator(self.opu, self.d, compute_calibration=(not self.light_memory),
                                                               use_calibration_transform=calibration_param_estimation,
                                                               encoding_decoding_precision=self.encoding_decoding_precision, use_torch=self.use_torch)
        self.calibration_param_estimation = calibration_param_estimation
        self.switch_use_calibration_forward = calibration_forward  # if True, use the calibrated OPU (the implicit matrix of the OPU) for the forward multiplication
        self.switch_use_calibration_backward = calibration_backward  # same, but for the gradient (backward) computation

        self.mu_opu, self.std_opu, self.norm_scaling = self.get_distribution_opu()
        # self.mu_opu, self.std_opu, self.norm_scaling = self.get_distribution_opu(light_memory=False)
        self.re_center_result = re_center_result
        # todoopu deplacer ca

    def get_distribution_opu(self):
        if self.calibration_param_estimation:
            mu = self.distribution_estimator.mu_estimation(method="mean")
            std = np.sqrt(self.distribution_estimator.var_estimation(method="var"))
            # multiplied by 1/std because we want the norm of the matrix
            # whose coefficients are sampled in N(0,1)
            col_norm = self.module_math_functions.linalg.norm(self.distribution_estimator.FHB * 1./std, axis=0)
            col_norm[self.module_math_functions.where(col_norm == 0)] = np.inf

        else:
            # todo choisir le n_iter dynamiquement et utiliser une autre methode que "ones"
            mu = self.distribution_estimator.mu_estimation(method="ones")
            std = np.sqrt(self.distribution_estimator.var_estimation(method="ones"))
            col_norm = np.sqrt(self.d) * self.module_math_functions.ones(self.opu.n_components)

        return mu, std, col_norm

    def init_shape(self):
        if isinstance(self.opu.device, SimulatedOpuDevice):
            d = self.opu.max_n_features
        else:
            d = self.provided_dimension

        m = self.opu.n_components
        if self.bool_multiple_sigmas:
            return (d,
                    m * len(self.SigFact))

        else:
            return (d, m)


    def set_use_calibration_forward(self, boolean):
        if boolean is True and self.light_memory is True:
            raise ValueError("Can't switch calibration ON when light_memory is True")
        self.switch_use_calibration_forward = boolean

    def set_use_calibration_backward(self, boolean):
        if boolean is True and self.light_memory is True:
            raise ValueError("Can't switch calibration ON when light_memory is True")
        self.switch_use_calibration_backward = boolean

    def _OPU(self, x):
        if self.switch_use_calibration_forward:
            if self.use_torch:
                if x.ndim == 1:
                    return LinearFunctionEncDec.apply(x.unsqueeze(0), self.calibrated_matrix).squeeze(0)
                else:
                    return LinearFunctionEncDec.apply(x, self.calibrated_matrix)
            else:
                return self.wrap_transform(lambda inp: inp @ self.calibrated_matrix, x, precision_encoding=self.encoding_decoding_precision)()
        else:
            # if self.light_memory:
            if self.use_torch:
                if x.ndim == 1:
                    return OPUFunctionEncDec.apply(x.unsqueeze(0), self.opu.linear_transform, self.calibrated_matrix,  self.encoding_decoding_precision).squeeze(0)
                else:
                    return OPUFunctionEncDec.apply(x, self.opu.linear_transform, self.calibrated_matrix, self.encoding_decoding_precision)
            else:
                return self.wrap_transform(self.opu.linear_transform, x, precision_encoding=self.encoding_decoding_precision)()

    def get_randn_mat(self, mu=0, sigma=1.):
        if self.light_memory is True:
            raise ValueError("The OPU implicit matrix is unknown when light_memory is True.")
        if self.re_center_result:
            return ((self.calibrated_matrix - self.mu_opu) * 1./self.std_opu) * sigma + mu
        else:
            return (self.calibrated_matrix * 1./self.std_opu) * sigma + mu

    @property
    def calibrated_matrix(self):
        return self.distribution_estimator.FHB

    def directions_matrix(self):
        return self.get_randn_mat() * 1. / self.norm_scaling
        # return (self.calibrated_matrix * 1. / self.std_opu * 1. / self.norm_scaling)

    def applyOPU(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.bool_sigfact_a_matrix:
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
        if self.re_center_result:
            mu_x = self.mu_opu * self.module_math_functions.sum(x, axis=1)  # sum other all dims
            y_dec = y_dec - mu_x.reshape(-1, 1)

        y_dec = self.R * y_dec * 1./self.std_opu * 1./self.norm_scaling
        if not self.bool_sigfact_a_matrix:
            if self.bool_multiple_sigmas:
                if self.use_torch:
                    y_dec = self.module_math_functions.einsum("ij,jkl->kil", self.SigFact.unsqueeze(-1), y_dec.unsqueeze(0))
                else:
                    y_dec = self.module_math_functions.einsum("ij,jkl->kil", self.SigFact[:, self.module_math_functions.newaxis], y_dec[self.module_math_functions.newaxis])
                y_dec = y_dec.reshape((x.shape[0], y_dec.shape[-1] * len(self.SigFact)))
            else:
                y_dec = y_dec * self.SigFact

        out = y_dec
        return out

    # call the FeatureMap object as a function
    def call(self, x):
        # return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
        if self.bool_multiple_sigmas:
            return self.c_norm * self.f(self.applyOPU(x) + self.xi)  # Evaluate the feature map at x
        else:
            return self.c_norm * self.f(self.applyOPU(x) + self.xi)  # Evaluate the feature map at x

    @property
    def Omega(self):
        if self._Omega is None:
            try:
                self._Omega = self.SigFact @ self.directions_matrix() * self.R
            except ValueError:
                self._Omega = self.SigFact * self.directions_matrix() * self.R
        return self._Omega

    def grad(self, x):
        if self.switch_use_calibration_backward:
            f_grad_val = self.f_grad(np.matmul(self.Omega.T, x.T).T + self.xi)
            new = self.c_norm * np.einsum("ij,kj->ikj", f_grad_val, self.Omega)
            if x.shape[0] == 1 or x.ndim == 1:
                return new.squeeze()
            else:
                return new
        else:
            raise NotImplementedError("OPU doesn't have a gradient.")


if __name__ == "__main__":
    opu = OPU(n_components=2, opu_device=SimulatedOpuDevice(),
              max_n_features=2)
    opu.fit1d(n_features=2)
    # Phi = OPUFeatureMap("ComplexExponential", opu)
    sample = np.random.rand(1, 2)
    enc_dec_fct(opu.linear_transform, sample)
    enc_dec_fct(opu.linear_transform, sample)

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
