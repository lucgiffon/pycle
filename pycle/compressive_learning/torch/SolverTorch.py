from abc import abstractmethod

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from pycle.compressive_learning.Solver import Solver
from pycle.utils.optim import ObjectiveValuesStorage


class SolverTorch(Solver):
    """
    Adapt Solver base methods to torch.
    """

    def __init__(self, phi, sketch, maxiter_inner_optimizations=1000, tol_inner_optimizations=1e-5,  beta_1=0., beta_2=0.99,
                 lr_inner_optimizations=0.01, show_curves=False, tensorboard=False, *args, **kwargs):

        self.maxiter_inner_optimizations = maxiter_inner_optimizations
        self.tol_inner_optimizations = tol_inner_optimizations
        self.lr_inner_optimizations = lr_inner_optimizations
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch.device.type
        self.device = phi.device
        self.real_dtype = phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        self.show_curves = show_curves
        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter()

        super().__init__(phi=phi, sketch=sketch, *args, **kwargs)

    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(self.current_sol)
            self.current_sol_cost = torch.norm(self.residual)
        else:
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf

    def initialize_empty_solution(self):
        self.n_atoms = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)
        self.all_atoms = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)
        self.current_sol = (self.alphas, self.all_thetas)  # Overwrite

    @abstractmethod
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError
