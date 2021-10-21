import torch
import numpy as np

from pycle.compressive_learning.Solver import Solver


class SolverTorch(Solver):
    """
    Adapt Solver base methods to torch.
    """

    def __init__(self, phi, sketch, *args, **kwargs):

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch.device.type
        self.device = phi.device
        self.real_dtype = phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

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