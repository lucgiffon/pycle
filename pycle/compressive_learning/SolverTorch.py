from abc import abstractmethod

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pycle.compressive_learning.Solver import Solver
from pycle.sketching import FeatureMap
from pycle.utils.optim import ObjectiveValuesStorage


# cleaning make documentation and clean everything here
class SolverTorch(Solver):
    """
    Adapt Solver base methods to torch.
    """

    def initialize_parameters_optimization(self) -> None:
        """
        Transform optimization parameters in dct_opt_method to actual attributes of the object.
        Further tests could be done here, as the adequation between the optimization method used and the parameters provided.
        :return:
        """
        # todo this function should not be aware of what happens in child classes
        self.maxiter_inner_optimizations = self.dct_opt_method.get("maxiter_inner_optimizations", 15000)
        self.tol_inner_optimizations = self.dct_opt_method.get("tol_inner_optimizations", 1e-9)
        self.lr_inner_optimizations = self.dct_opt_method.get("lr_inner_optimizations", 1)
        self.beta_1 = self.dct_opt_method.get("beta_1", 0.9)
        self.beta_2 = self.dct_opt_method.get("beta_2", 0.99)
        self.nb_iter_max_step_5 = self.dct_opt_method.get("nb_iter_max_step_5", 200)
        self.nb_iter_max_step_1 = self.dct_opt_method.get("nb_iter_max_step_1", 200)

        self.opt_method_step_1 = self.dct_opt_method.get("opt_method_step_1", "lbfgs")
        self.opt_method_step_34 = self.dct_opt_method.get("opt_method_step_34", "nnls")
        self.opt_method_step_5 = self.dct_opt_method.get("opt_method_step_5", "lbfgs")

        self.lambda_l1 = self.dct_opt_method.get("lambda_l1", 0)


    def __init__(self, phi: FeatureMap, sketch,
                 show_curves: bool = False, tensorboard: bool = False,
                 path_template_tensorboard_writer="CLOMP/{}/loss/", dct_opt_method: [None, dict] = None,
                 *args, **kwargs):

        # Attributes related to the optimization method used
        self.dct_opt_method = dct_opt_method or dict()  # todo utiliser le dicitonnaire d'optim
        self.initialize_parameters_optimization()

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch.device.type
        self.device = phi.device
        self.real_dtype = phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        # parameters regarding the tracking of the algorithm
        self.show_curves = show_curves
        self.tensorboard = tensorboard
        self.path_template_tensorboard_writer = path_template_tensorboard_writer
        if tensorboard:
            self.writer = SummaryWriter()

        super().__init__(phi=phi, sketch=sketch, *args, **kwargs)

    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution.
        If `sol` given, also updates the `current_sol` attribute."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(all_thetas=self.all_thetas, alphas=self.alphas)
            self.current_sol_cost = torch.norm(self.residual) ** 2
        else:
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf

    def initialize_empty_solution(self) -> None:
        """
        Attributes pertaining to the solution are :

         - `n_atoms`, `alphas`, `all_thetas`, `all_atoms`, `residual` and `current_sol`.

        """
        self.n_atoms = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)
        self.all_atoms = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)
        self.current_sol = (self.all_thetas, self.alphas)  # Overwrite

    @abstractmethod
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError
