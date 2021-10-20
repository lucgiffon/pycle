from abc import ABCMeta

import numpy as np

from pycle.compressive_learning.Solver import Solver


class SolverNumpy(Solver, metaclass=ABCMeta):
    """
    Adapt Solver base methods to numpy.
    """
    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(self.current_sol)
            self.current_sol_cost = np.linalg.norm(self.residual)
        else:
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf