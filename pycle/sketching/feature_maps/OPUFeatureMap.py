"""
This module contains functions and classes related to the use of the OPU to compute the random fourier features
in the context of sketching.
"""

from typing import Callable, Literal, Union, Optional, Tuple, NoReturn

import numpy as np
import torch
from fht import fht
from lightonml import OPU
from lightonml.internal.simulated_device import SimulatedOpuDevice
from scipy.linalg import hadamard

from pycle.sketching import FeatureMap
from pycle.sketching.distribution_estimation import mu_estimation_ones, var_estimation_ones, var_estimation_randn, \
    var_estimation_any
from pycle.utils import is_number
from pycle.utils.torch_functions import LinearFunctionEncDec, OPUFunctionEncDec
from pycle.utils.encoding_decoding import enc_dec_fct


def calibrate_lin_op(fct_lin_op: Callable, dim_in: int, nb_iter: int = 1) -> np.ndarray:
    """
    Perform calibration of a linear operator with input dimension dim.

    This works by applying the linear operator the the Hadamard matrix then solving all the linear equations
    to recover the corresponding columns of the linear operator.

    Notes
    -----
    - The `fct_lin_op` parameter must take row vectors as input.

    Parameters
    ----------
    fct_lin_op
        The linear operator function to call with one 2D np-array
    dim_in
        input dimension of linear operator
    nb_iter
        if the linear operator has centered noise, result will be the average of nb_iter runs

    Returns
    -------
    np.ndarray
        The matrix corresponding to the calibrated linear operator.
    """
    first_pow_of_2_gt_d = 2 ** int(np.ceil(np.log2(dim_in)))
    H = hadamard(first_pow_of_2_gt_d)
    # todo make torch compatible:
    #  problem is that fct_lin_op must be able to take numpy arr as input which won't always be the case
    H_truncated_left = H[:dim_in, :dim_in]
    # keep only the first dim rows to pretend fct_lin_op have been padded with zeros
    H_truncated_right = H[:dim_in, dim_in:]
    # the last cols of H need also to be transformed by fct_lin_op so that the output result has the right shape
    # for backward FHT.

    def compute_B():
        # the transpose is because fct_lin_op transforms rows and not cols
        # because hadamard is symmetric, it has no effect on the left side (also symmetric),
        # but I write it for lisibility
        acc_B_truncated_left = np.array(
            fct_lin_op(H_truncated_left.T > 0) - fct_lin_op(H_truncated_left.T < 0))

        if bool(H_truncated_right.size):
            acc_B_truncated_right = np.array(
                fct_lin_op(H_truncated_right.T > 0) - fct_lin_op(H_truncated_right.T < 0))
        else:
            assert first_pow_of_2_gt_d == dim_in
            acc_B_truncated_right = np.empty((0, acc_B_truncated_left.shape[1]))

        return acc_B_truncated_left, acc_B_truncated_right

    def make_one_iteration():
        acc_B_truncated_left, acc_B_truncated_right = compute_B()
        # this B is the result of W H as if W was padded with cols full of zeros. W stands for the matrix repr of fct_lin_op
        # B = np.array((self.opu.linear_transform(H > 0) - self.opu.linear_transform(H < 0)))

        B = np.vstack([acc_B_truncated_left, acc_B_truncated_right])
        sqrt_d = np.sqrt(first_pow_of_2_gt_d)  # need to rescale the result because fht implements a scaled Hadamard matrix
        FHB = np.array([1. / first_pow_of_2_gt_d * fht(b) * sqrt_d for b in B.T]).T
        # FHB = H @ B / self.d
        return FHB[:dim_in]

    FHB = make_one_iteration()
    i_iter = 1
    while i_iter < nb_iter:
        FHB += make_one_iteration()
        i_iter += 1

    return FHB / nb_iter


class OPUDistributionEstimator:
    def __init__(self, opu: OPU, input_dim: int,
                 compute_calibration: bool = False, use_calibration_transform: bool = False,
                 nb_iter_calibration: int = 1, encoding_decoding_precision: int = 8,
                 device: torch.device = torch.device("cpu")):
        """
        Class to make estimations on the distribution of the implicit OPU matrix.
        Also capable to calibrate the implicit OPU matrix.

        Parameters
        ----------
        opu:
            The OPU object to perform the distribution estimation on.
        input_dim:
            The input dimension.
        compute_calibration:
            Calibrate the OPU and store it in memory.
        use_calibration_transform:
            Use calibration to make distribution estimation. If True, `compute_calibration` must also be True.
        nb_iter_calibration:
            Average the calibration on `nb_iter_calibration` estimates.
        encoding_decoding_precision:
            Number of precision bits when calling the OPU.
        device:
            Device on which to perform torch operations.
        """
        self.opu = opu
        self.d = input_dim
        self.m = self.opu.n_components
        self.device = device

        self.use_calibration_transform = use_calibration_transform
        self.compute_calibration = compute_calibration
        self.nb_iter_calibration = nb_iter_calibration
        # if calibration is used for distribution estimation, then it must have been computed first.
        assert use_calibration_transform is False or compute_calibration is True
        if self.compute_calibration:
            self.FHB = self.calibrate_opu().to(self.device)
        else:
            self.FHB = None

        self.encoding_decoding_precision = encoding_decoding_precision

    def OPU(self, x: torch.Tensor) -> torch.Tensor:
        """
        A handler function allowing to call the OPU wrapped with encoding and decoding functions.

        Parameters
        ----------
        x
            The input to apply the OPU to.

        Returns
        -------
            The output of the OPU applied to x.
        """
        return enc_dec_fct(self.opu.linear_transform, x, precision_encoding=self.encoding_decoding_precision)

    def transform(self, x: torch.Tensor, direct=False) -> torch.Tensor:
        """
        A handler function to make the linear transformation of the OPU using the calibrated matrix or the real OPU.

        Parameters
        ----------
        x
            The input to transform.
        direct
            If True and the real OPU is used, transforms x without encoding/decoding wrapping.

        Returns
        -------
            Returns the output of the OPU applied to the input.
        """
        assert 0 < x.ndim <= 2
        if self.use_calibration_transform:
            y = x @ self.FHB
        else:
            if direct:
                y = self.opu.linear_transform(x)
            else:
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                y = self.OPU(x)

        return y

    def calibrate_opu(self) -> torch.Tensor:
        """
        Get the calibrated transmission matrix of the OPU.

        Returns
        -------
            The OPU transmission matrix.
        """
        # calibrate_lin_op works with numpy arrays,
        # it means that self.opu.linear transform must always be able to take numpy arrays as input
        return torch.from_numpy(calibrate_lin_op(lambda x: self.opu.linear_transform(x), self.d,
                                                 nb_iter=self.nb_iter_calibration))

    def mu_estimation(self, method: Literal["ones", "mean"] = "ones") -> torch.float:
        """
        Estimate the mean of the coefficients in the OPU transmission matrix.

        Parameters
        ----------
        method
            "ones" to probe with a vector of ones. "mean" to use the calibrated transmission matrix.

        Returns
        -------
            The mean value of the implicit OPU transmission matrix.
        """
        if method == "ones":
            return mu_estimation_ones(lambda x: self.transform(x, direct=True), self.d)
        elif method == "mean":
            try:
                return torch.mean(self.FHB)
            except TypeError as e:  # FHB is None
                raise ValueError(f"Method `mean` only works with `compute_calibration` flag.")
        else:
            raise ValueError(f"Unknown method: {method}.")

    def var_estimation(self, method: Literal["var", "any", "randn", "ones"] = "var", n_iter: int = 1) -> torch.float:
        """
        Estimate the variance of the coefficients in the OPU transmission matrix.

        Parameters
        ----------
        method
            "ones" to probe with a vector of ones; "var" to use the calibrated transmission matrix;
            "any" to use any vectors of known norm; "randn" to use vectors sampled from gaussian distribution.
        n_iter
            Number of iterations for methods "any" and "randn". No effect otherwise.

        Returns
        -------
            The variance of the coefficients in the implicit OPU transmission matrix.
        """
        if method == "ones":
            return var_estimation_ones(lambda x: self.transform(x, direct=True),
                                       self.d)
        elif method == "any":
            return var_estimation_any(lambda x: self.transform(x),
                                      self.d, n_iter)
        elif method == "randn":
            return var_estimation_randn(lambda x: self.transform(x),
                                        self.d, n_iter)
        elif method == "var":
            try:
                return torch.var(self.FHB)
            except TypeError:  # FHB is None
                raise ValueError(f"Method `var` only works with `compute_calibration` parameter of {self.__class__.__name__} set to True.")
        else:
            raise ValueError(f"Unknown method: {method}.")


class OPUFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(OPU(x) + xi)."""

    def __init__(self, f: Optional[Union[Literal["complexexponential", "universalquantization", "cosine", "none"], Callable]],
                 opu: OPU, SigFact: Union[torch.FloatTensor, torch.Tensor], R: torch.Tensor, dimension: int,
                 nb_iter_calibration: int = 1, nb_iter_linear_transformation: int = 1,
                 calibration_param_estimation: Optional[bool] = None, calibration_forward: bool = False,
                 calibrate_always: bool = False,
                 device: torch.device = torch.device("cpu"), **kwargs):
        """
        Random Fourier Features feature map based on an OPU to perform the random transformation.

        Parameters
        ----------
        f:
            The activation function for the feature map. Default: "complexexponential".
        opu:
            The OPU obect used for linear transformations.
        SigFact:
            The sigma factor for the frequency sampling scheme.
        R:
            The vector of radii for each frequencies.
        dimension:
            The input dimension.
        nb_iter_calibration:
            The number of iterations for the calibration averaging.
        nb_iter_linear_transformation:
            The number of iterations for each linear transformation with the OPU.
        calibration_param_estimation:
            Tells if calibration should be used to evaluate the parameters of the OPU transmission matrix.
        calibration_forward:
            Tells if calibration should be used for the feature map computation.
        calibrate_always:
            Tells if calibration should be done anyway.
        device:
            The device on which to perform the tensor operations. torch.device("cpu") or torch.device("cuda:\*").
        kwargs:
            Other key word arguments for FeatureMap object.
        """
        self.opu = opu
        self.dimension = dimension

        if isinstance(self.opu.device, SimulatedOpuDevice):
            assert self.opu.max_n_features == self.dimension

        self.SigFact = SigFact.to(device)
        self.bool_sigfact_a_matrix = isinstance(self.SigFact, torch.Tensor) and self.SigFact.ndim > 1

        self._Omega = None
        self.R = R.to(device)
        if self.R.ndim == 1:
            self.R = self.R.unsqueeze(-1)

        super().__init__(f, dtype=self.Omega_dtype, device=device, **kwargs)

        assert self.R.shape[0] == self.opu.n_components
        if is_number(self.SigFact):
            self.SigFact = torch.Tensor([self.SigFact]).to(self.device)

        self.light_memory = (not (calibration_param_estimation or calibration_forward)) and (not calibrate_always)
        if calibration_param_estimation is None and not self.light_memory:
            calibration_param_estimation = True
        else:
            calibration_param_estimation = calibration_param_estimation or False
        self.nb_iter_calibration = nb_iter_calibration
        self.nb_iter_linear_transformation = nb_iter_linear_transformation

        self.distribution_estimator = OPUDistributionEstimator(self.opu, self.d,
                                                               compute_calibration=(not self.light_memory),
                                                               use_calibration_transform=calibration_param_estimation,
                                                               nb_iter_calibration=self.nb_iter_calibration,
                                                               encoding_decoding_precision=self.encoding_decoding_precision,
                                                               device=self.device)
        self.calibration_param_estimation = calibration_param_estimation
        self.switch_use_calibration_forward = calibration_forward

        # mu_opu should be around zero
        self.mu_opu, self.std_opu, self.norm_scaling = self.get_distribution_opu()

    def get_distribution_opu(self) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get the mean, standard deviation and norm of the lines of the OPU transmission matrix.

        Returns
        -------
            mean, std, norm
        """
        if self.calibration_param_estimation:
            mu = self.distribution_estimator.mu_estimation(method="mean").to(self.device)
            std = torch.sqrt(self.distribution_estimator.var_estimation(method="var")).to(self.device)
            # multiplied by 1/std because we want the norm of the matrix
            # whose coefficients are sampled in N(0,1)
            col_norm = torch.linalg.norm(self.distribution_estimator.FHB * 1./std, axis=0).to(self.device)
            col_norm[torch.where(col_norm == 0)] = np.inf

        else:
            # todo choisir le n_iter dynamiquement et utiliser une autre methode que "ones"
            mu = self.distribution_estimator.mu_estimation(method="ones").to(self.device)
            std = torch.sqrt(self.distribution_estimator.var_estimation(method="ones")).to(self.device)
            col_norm = np.sqrt(self.d) * torch.ones(self.opu.n_components).to(self.device)

        return mu, std, col_norm

    def init_shape(self) -> Tuple[int, int]:
        """
        The shape of the linear transformation matrix used inside the feature map.

        The shape correspond to the (input, output) dimensions.

        Returns
        -------
            The (input, output) dimensions of the feature map.
        """
        d = self.dimension

        m = self.opu.n_components
        if not self.bool_sigfact_a_matrix:
            m = m * len(self.SigFact)
        m = m * self.R.shape[1]
        return d, m

    @property
    def Omega_dtype(self) -> torch.dtype:
        """
        Returns
        -------
            The type of the Omega linear transform matrix
        """
        return torch.promote_types(self.R.dtype, self.SigFact.dtype)

    def set_use_calibration_forward(self, boolean: bool) -> NoReturn:
        """
        Method to set the toggle for calibration use for the feature map application.

        Parameters
        ----------
        boolean:
            If True, toggle ON.
        """
        if boolean is True and self.light_memory is True:
            raise ValueError("Can't switch calibration ON when light_memory is True")
        self.switch_use_calibration_forward = boolean

    def _OPU(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the OPU wrapped in a torch function to the input tensor.

        Parameters
        ----------
        x:
            Input observations as rows of a tensor.

        Returns
        -------
            result
        """
        if self.switch_use_calibration_forward:
            if x.ndim == 1:
                return LinearFunctionEncDec.apply(x.unsqueeze(0), self.calibrated_matrix).squeeze(0)
            else:
                return LinearFunctionEncDec.apply(x, self.calibrated_matrix)
        else:
            if x.ndim == 1:
                return OPUFunctionEncDec.apply(x.unsqueeze(0).cpu(), self.opu.linear_transform, self.calibrated_matrix,
                                               self.encoding_decoding_precision,
                                               self.nb_iter_linear_transformation).squeeze(0).to(self.device)
            else:
                return OPUFunctionEncDec.apply(x.cpu(), self.opu.linear_transform, self.calibrated_matrix,
                                               self.encoding_decoding_precision,
                                               self.nb_iter_linear_transformation).to(self.device)

    def get_randn_mat(self, mu: float = 0, sigma: float = 1.) -> torch.Tensor:
        """
        Get the transmission matrix of the OPU centered on `mu` and reduced to `sigma` scale.

        Parameters
        ----------
        mu:
            The target mean.
        sigma:
            The target variance.

        Returns
        -------
            Calibrated transmission matrix centered and rescaled.
        """
        if self.light_memory is True:
            raise ValueError("The OPU implicit matrix is unknown when light_memory is True.")
        return (self.calibrated_matrix * 1./self.std_opu) * sigma + mu

    @property
    def calibrated_matrix(self) -> torch.Tensor:
        """
        Returns
        -------
            The calibrated transmission matrix of the OPU.
        """
        return self.distribution_estimator.FHB

    def directions_matrix(self) -> torch.Tensor:
        """
        Get the transmission matrix with normalized rows such that each row is like sampled uniformly on
        the unit sphere.

        Returns
        -------
            The matrix of unit norm directions.
        """
        r = self.get_randn_mat() * 1. / self.norm_scaling
        r = r.to(self.dtype).to(self.device)
        return r

    def lin_op_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Application of the OPU to the input.

        Parameters
        ----------
        x:
            Input observations as rows of a tensor.

        Returns
        -------
            result
        """
        return self.applyOPU(x)

    def applyOPU(self, x: torch.Tensor) -> torch.Tensor:
        """
        Application of the OPU to the input.

        Parameters
        ----------
        x:
            Input observations as rows of a tensor.

        Returns
        -------
            result
        """
        reshaped_1_ndim = False
        if x.ndim == 1:
            reshaped_1_ndim = True
            x = x.reshape(1, -1)

        if self.bool_sigfact_a_matrix:
            x = x @ self.SigFact

        y_dec = self._OPU(x)

        y_dec = y_dec * 1./self.std_opu

        y_dec = (y_dec * 1./self.norm_scaling).unsqueeze(-1) * self.R

        if not self.bool_sigfact_a_matrix:
            y_dec = torch.einsum("ijk,h->ikhj", y_dec, self.SigFact)
            y_dec = y_dec.reshape((x.shape[0], self.m))

        out = y_dec
        if reshaped_1_ndim:
            out = out.squeeze()
        return out

    @property
    def Omega(self) -> torch.Tensor:
        """
        Returns
        -------
            The frequencies matrix obtained from the OPU transmission matrix
        """
        if self._Omega is None:
            try:
                self._Omega = self.SigFact @ self.directions_matrix() * self.R
            except:  # if SigFact is not a matrix
                self._Omega = (self.SigFact * self.directions_matrix()).unsqueeze(-1) * self.R
                self._Omega = torch.reshape(self._Omega, (-1, self.m))
        return self._Omega
